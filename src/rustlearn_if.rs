use crate::error::AARCError;
use lazy_static::*;
use log::{info, trace};
use num_cpus;
use obj_pool::ObjPool;
use rustlearn::cross_validation::CrossValidation;
use rustlearn::ensemble::random_forest;
use rustlearn::factorization::factorization_machines;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics::accuracy_score;
use rustlearn::multiclass::OneVsRestWrapper;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree;
use std::sync::Mutex;

lazy_static! {
    static ref POOL: Mutex<ObjPool<OneVsRestWrapper<sgdclassifier::SGDClassifier>>> =
        Mutex::new(ObjPool::new());
    static ref RFPOOL: Mutex<ObjPool<OneVsRestWrapper<random_forest::RandomForest>>> =
        Mutex::new(ObjPool::new());
    static ref FACPOOL: Mutex<ObjPool<OneVsRestWrapper<factorization_machines::FactorizationMachine>>> =
        Mutex::new(ObjPool::new());
    static ref NUM_CPUS: usize = num_cpus::get();
}

pub fn clear_objects() {
    let mut pool = POOL.lock().unwrap();
    pool.clear();
    let mut pool = RFPOOL.lock().unwrap();
    pool.clear();
}

pub fn train_logistic_regression(
    input: &Array,
    train: &Array,
    learning_rate: f64,
    l1_penalty: f64,
    l2_penalty: f64,
    cross_fold: usize,
    epochs: usize,
) -> Result<Vec<f64>, AARCError> {
    info!("Creating logistic regression model");
    let mut model = sgdclassifier::Hyperparameters::new(input.data().len() / train.data().len())
        .learning_rate(learning_rate as f32)
        .l1_penalty(l1_penalty as f32)
        .l2_penalty(l2_penalty as f32)
        .one_vs_rest();
    let mut accuracy = 0.0;

    for (train_idx, test_idx) in CrossValidation::new(input.rows(), cross_fold) {
        let x_train = input.get_rows(&train_idx);
        let y_train = train.get_rows(&train_idx);
        let x_test = input.get_rows(&test_idx);
        let y_test = train.get_rows(&test_idx);

        // Save the model in memory, for later prediction
        trace!("Fitting model {} times", epochs);
        for _ in 0..epochs {
            match model.fit_parallel(&x_train, &y_train, *NUM_CPUS) {
                Ok(m) => trace!("Learnt something {:?}", m),
                Err(e) => return Err(AARCError::LogisticRegressionError(e.to_string())),
            }
        }
        // Predict and get accuracy score
        let prediction = model.predict_parallel(&x_test, *NUM_CPUS).unwrap();
        accuracy += accuracy_score(&y_test, &prediction);
        trace!("Accuracy {}", accuracy / cross_fold as f32);
    }
    let mut pool = POOL.lock().unwrap();
    let id = pool.insert(model);
    Ok(vec![
        pool.obj_id_to_index(id) as f64,
        (accuracy / cross_fold as f32) as f64,
    ])
}

pub fn predict_logistic_regression(id: f64, predict: Array) -> Result<Array, AARCError> {
    let pool = POOL.lock().unwrap();
    let obj_id = pool.index_to_obj_id(id as u32);
    let model = pool
        .get(obj_id)
        .ok_or_else(|| AARCError::ModelNotFoundError(id as u32))?;
    info!(
        "Predicting using Regression model {} {}",
        id,
        predict.data().len()
    );
    model
        .predict_parallel(&predict, *NUM_CPUS)
        .map_err(|e| AARCError::LogisticRegressionError(e.to_string()))
}

pub fn train_random_forest(
    input: &Array,
    train: &Array,
    num_trees: usize,
    min_samples_split: usize,
    max_features: usize,
    max_depth: usize,
    cross_fold: usize,
    epochs: usize,
) -> Result<Vec<f64>, AARCError> {
    trace!(
        "Configuring tree parameters min_samples {} max_depth:{} max_feature:{}",
        min_samples_split,
        max_depth,
        max_features
    );
    let mut tree = decision_tree::Hyperparameters::new(input.cols());
    tree.min_samples_split(min_samples_split)
        .max_depth(max_depth)
        .max_features(max_features);
    info!("Configuring random forest");
    let mut model = random_forest::Hyperparameters::new(tree, num_trees).one_vs_rest();
    let mut accuracy = 0.0;

    for (train_idx, test_idx) in CrossValidation::new(input.rows(), cross_fold) {
        let x_train = input.get_rows(&train_idx);
        let y_train = train.get_rows(&train_idx);
        let x_test = input.get_rows(&test_idx);
        let y_test = train.get_rows(&test_idx);

        // Save the model in memory, for later prediction
        trace!("Fitting model {} times", epochs);
        for _ in 0..epochs {
            match model.fit_parallel(&x_train, &y_train, *NUM_CPUS) {
                Ok(m) => trace!("Learnt something {:?}", m),
                Err(e) => return Err(AARCError::LogisticRegressionError(e.to_string())),
            }
        }
        // Predict and get accuracy score
        let prediction = model.predict_parallel(&x_test, *NUM_CPUS).unwrap();
        accuracy += accuracy_score(&y_test, &prediction);
        trace!("Accuracy {}", accuracy / cross_fold as f32);
    }
    let mut pool = RFPOOL.lock().unwrap();
    let id = pool.insert(model);
    Ok(vec![
        pool.obj_id_to_index(id) as f64,
        (accuracy / cross_fold as f32) as f64,
    ])
}

pub fn predict_random_forest(id: f64, predict: Array) -> Result<Array, AARCError> {
    let pool = RFPOOL.lock().unwrap();
    let obj_id = pool.index_to_obj_id(id as u32);
    let model = pool
        .get(obj_id)
        .ok_or_else(|| AARCError::ModelNotFoundError(id as u32))?;
    info!(
        "Predicting using Random Forest model {} {}",
        id,
        predict.data().len()
    );
    model
        .predict_parallel(&predict, *NUM_CPUS)
        .map_err(|e| AARCError::LogisticRegressionError(e.to_string()))
}

pub fn train_factorization(
    input: &Array,
    train: &Array,
    num_components: usize,
    learning_rate: f64,
    l1_penalty: f64,
    l2_penalty: f64,
    cross_fold: usize,
    epochs: usize,
) -> Result<Vec<f64>, AARCError> {
    info!("Training factorization model");
    let mut model = factorization_machines::Hyperparameters::new(
        input.data().len() / train.data().len(),
        num_components,
    )
    .learning_rate(learning_rate as f32)
    .l1_penalty(l1_penalty as f32)
    .l2_penalty(l2_penalty as f32)
    .one_vs_rest();
    let mut accuracy = 0.0;

    for (train_idx, test_idx) in CrossValidation::new(input.rows(), cross_fold) {
        let x_train = input.get_rows(&train_idx);
        let y_train = train.get_rows(&train_idx);
        let x_test = input.get_rows(&test_idx);
        let y_test = train.get_rows(&test_idx);

        // Save the model in memory, for later prediction
        trace!("Fitting model {} times", epochs);
        for _ in 0..epochs {
            match model.fit_parallel(&x_train, &y_train, *NUM_CPUS) {
                Ok(m) => trace!("Learnt something {:?}", m),
                Err(e) => return Err(AARCError::LogisticRegressionError(e.to_string())),
            }
        }
        // Predict and get accuracy score
        let prediction = model.predict_parallel(&x_test, *NUM_CPUS).unwrap();
        accuracy += accuracy_score(&y_test, &prediction);
        trace!("Accuracy {}", accuracy / cross_fold as f32);
    }
    let mut pool = FACPOOL.lock().unwrap();
    let id = pool.insert(model);
    Ok(vec![
        pool.obj_id_to_index(id) as f64,
        (accuracy / cross_fold as f32) as f64,
    ])
}

pub fn predict_factorization(id: f64, predict: Array) -> Result<Array, AARCError> {
    let pool = FACPOOL.lock().unwrap();
    let obj_id = pool.index_to_obj_id(id as u32);
    let model = pool
        .get(obj_id)
        .ok_or_else(|| AARCError::ModelNotFoundError(id as u32))?;
    info!(
        "Predicting Factorization model {} {}",
        id,
        predict.data().len()
    );
    model
        .predict_parallel(&predict, *NUM_CPUS)
        .map_err(|e| AARCError::LogisticRegressionError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::debug;
    use rustlearn::datasets::iris;
    use simplelog::{Config, LevelFilter, TermLogger};

    #[test]
    fn test_rust_learn() {
        TermLogger::init(LevelFilter::Trace, Config::default()).unwrap();
        debug!("Logging started.");
        let (X, y) = iris::load_data();
        for i in 0..X.data().len() / 4 {
            println!(
                "{},{},{},{}",
                X.data()[i],
                X.data()[i + 1],
                X.data()[i + 2],
                X.data()[i + 3]
            );
        }
        let num_splits = 1;
        let num_epochs = 5;

        let mut accuracy = 0.0;

        // for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {
        //     let X_train = X.get_rows(&train_idx);
        //     let y_train = y.get_rows(&train_idx);
        //     let X_test = X.get_rows(&test_idx);
        //     let y_test = y.get_rows(&test_idx);

        let mut model = Hyperparameters::new(X.cols())
            .learning_rate(0.5)
            .l2_penalty(0.0)
            .l1_penalty(0.0)
            .one_vs_rest();

        for _ in 0..num_epochs {
            model.fit(&X, &y).unwrap();
        }

        let prediction = model.predict(&X).unwrap();
        trace!("Prediction {:?}", prediction);
        accuracy += accuracy_score(&y, &prediction);
        // }
        debug!(
            "accuracy {} num_splits {} Overall:{}",
            accuracy,
            num_splits,
            accuracy / num_splits as f32
        );

        accuracy /= num_splits as f32;
    }

    #[test]
    fn test_excel_if() {
        TermLogger::init(LevelFilter::Trace, Config::default()).unwrap();
        debug!("Logging started.");
        let (X, y) = iris::load_data();
        let id = train_logistic_regression(&X, &y, 0.5, 0.0, 0.0, 10);
        let prediction = predict_logistic_regression(id, X);
    }
}
