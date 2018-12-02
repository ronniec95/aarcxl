use crate::error::AARCError;
use crate::normalize;
use crate::rustlearn_if;
use log::trace;
use rustlearn::prelude::*;
use std::convert::From;
use std::convert::TryInto;
use std::f64;
use xladd::variant::Variant;
// PDF from prices
// Random forest train/predict
// portfolio sim with vol and drift from pdf

pub fn normalize(
    array: Variant,
    min: Variant,
    max: Variant,
    norm_type: Variant,
) -> Result<Variant, AARCError> {
    let min: f64 = min.try_into()?;
    let max: f64 = max.try_into()?;
    let norm_type: f64 = norm_type.try_into()?;
    let (x, y) = array.dim();
    let array: Vec<f64> = array.into();
    let result = match norm_type as i64 {
        1 => normalize::tanh_est(&array),
        _ => normalize::min_max_norm(&array, min, max),
    };
    Ok(Variant::as_float_array(result, x, y))
    // Zscore normalization
    // Tanh Normalization
}

pub fn clear_objects() {
    rustlearn_if::clear_objects();
}

pub fn train_logistic_reg(
    input: Variant,
    train: Variant,
    learning_rate: Variant,
    l1_penalty: Variant,
    l2_penalty: Variant,
    cross_fold: Variant,
    epochs: Variant,
) -> Result<Variant, AARCError> {
    let learning_rate: f64 = learning_rate.try_into().unwrap_or(0.5);
    let l1_penalty: f64 = l1_penalty.try_into().unwrap_or(0.0);
    let l2_penalty: f64 = l2_penalty.try_into().unwrap_or(0.0);
    let cross_fold: f64 = cross_fold.try_into().unwrap_or(2.0);
    let epochs: f64 = epochs.try_into().unwrap_or(2.0);
    let (input_cols, input_rows) = input.dim();
    let input = {
        if input_cols == 0 || input_rows == 0 {
            return Err(AARCError::PredictionDataSizeZero);
        }
        let v: Vec<f32> = input.into();
        trace!(
            "Input rows:{} cols:{} Total:{}",
            input_cols,
            input_rows,
            v.len()
        );
        let mut input = Array::from(v);
        input.reshape(input_rows, input_cols);
        input
    };
    let train = {
        let (_train_cols, train_rows) = train.dim();
        if train_rows == 0 || (train_rows != input_rows) {
            return Err(AARCError::PredictionDataSizeZero);
        }
        let v: Vec<f32> = train.into();
        trace!("Training rows:{} cols:1 Total:{}", train_rows, v.len());
        let mut train = Array::from(v);
        train.reshape(train_rows, 1);
        train
    };

    trace!(
        "LearningRate {} L1:{} L2: {} Crossfold:{} Epochs:{} ",
        learning_rate,
        l1_penalty,
        l2_penalty,
        cross_fold,
        epochs,
    );

    match rustlearn_if::train_logistic_regression(
        &input,
        &train,
        learning_rate,
        l2_penalty,
        l1_penalty,
        cross_fold as usize,
        epochs as usize,
    ) {
        Ok(v) => Ok(Variant::as_float_array(v, 2, 1)),
        Err(e) => Err(e),
    }
}

pub fn predict_logistic_reg(id: Variant, predict: Variant) -> Result<Variant, AARCError> {
    let id: f64 = id.try_into()?;
    let (predict_cols, predict_rows) = predict.dim();
    if predict_rows == 0 || predict_cols == 0 {
        return Err(AARCError::PredictionDataSizeZero);
    }
    let predict = {
        let v: Vec<f32> = predict.into();
        trace!(
            "Predict rows:{} cols:{} Total:{}",
            predict_cols,
            predict_rows,
            v.len()
        );
        let mut predict = Array::from(v);
        predict.reshape(predict_rows, predict_cols);
        predict
    };
    match rustlearn_if::predict_logistic_regression(id, predict) {
        Ok(v) => {
            let v = v
                .data()
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            trace!("Returning {:?}", v);
            let height = v.len();
            Ok(Variant::as_float_array(v, 1, height))
        }
        Err(e) => Err(e),
    }
}

pub fn train_random_forest(
    input: Variant,
    train: Variant,
    num_trees: Variant,
    min_samples: Variant,
    max_features: Variant,
    max_depth: Variant,
    cross_fold: Variant,
    epochs: Variant,
) -> Result<Variant, AARCError> {
    let num_trees: f64 = num_trees.try_into().unwrap_or(10.0);
    let min_samples: f64 = min_samples.try_into().unwrap_or(2.0);
    let max_depth: f64 = max_depth.try_into().unwrap_or(40.0);
    let cross_fold: f64 = cross_fold.try_into().unwrap_or(2.0);
    let epochs: f64 = epochs.try_into().unwrap_or(2.0);
    let (input_cols, input_rows) = input.dim();
    if input_cols == 0 || input_rows == 0 {
        return Err(AARCError::PredictionDataSizeZero);
    }
    let max_features: f64 = max_features.try_into().unwrap_or(input_cols as f64);
    let input = {
        let v: Vec<f32> = input.into();
        trace!(
            "Input rows:{} cols:{} Total:{}",
            input_cols,
            input_rows,
            v.len()
        );
        let mut input = Array::from(v);
        input.reshape(input_rows, input_cols);
        input
    };
    let train = {
        let (_train_cols, train_rows) = train.dim();
        if train_rows == 0 || (train_rows != input_rows) {
            return Err(AARCError::PredictionDataSizeZero);
        }
        let v: Vec<f32> = train.into();
        trace!("Training rows:{} cols:1 Total:{}", train_rows, v.len());
        let mut train = Array::from(v);
        train.reshape(train_rows, 1);
        train
    };

    trace!(
        "num_trees {} min samples:{} min_features: {} Crossfold:{} Epochs:{} ",
        num_trees,
        min_samples,
        max_features,
        cross_fold,
        epochs,
    );

    match rustlearn_if::train_random_forest(
        &input,
        &train,
        num_trees as usize,
        min_samples as usize,
        max_features as usize,
        max_depth as usize,
        cross_fold as usize,
        epochs as usize,
    ) {
        Ok(v) => Ok(Variant::as_float_array(v, 2, 1)),
        Err(e) => Err(e),
    }
}

pub fn predict_random_forest(id: Variant, predict: Variant) -> Result<Variant, AARCError> {
    let id: f64 = id.try_into()?;
    let (predict_cols, predict_rows) = predict.dim();
    if predict_rows == 0 || predict_cols == 0 {
        return Err(AARCError::PredictionDataSizeZero);
    }
    let predict = {
        let v: Vec<f32> = predict.into();
        trace!(
            "Predict rows:{} cols:{} Total:{}",
            predict_cols,
            predict_rows,
            v.len()
        );
        let mut predict = Array::from(v);
        predict.reshape(predict_rows, predict_cols);
        predict
    };
    match rustlearn_if::predict_random_forest(id, predict) {
        Ok(v) => {
            let v = v
                .data()
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            trace!("Returning {:?}", v);
            let height = v.len();
            Ok(Variant::as_float_array(v, 1, height))
        }
        Err(e) => Err(e),
    }
}

pub fn train_factorization(
    input: Variant,
    train: Variant,
    num_components: Variant,
    learning_rate: Variant,
    l1_penalty: Variant,
    l2_penalty: Variant,
    cross_fold: Variant,
    epochs: Variant,
) -> Result<Variant, AARCError> {
    let learning_rate: f64 = learning_rate.try_into().unwrap_or(0.5);
    let l1_penalty: f64 = l1_penalty.try_into().unwrap_or(0.0);
    let l2_penalty: f64 = l2_penalty.try_into().unwrap_or(0.0);
    let cross_fold: f64 = cross_fold.try_into().unwrap_or(2.0);
    let epochs: f64 = epochs.try_into().unwrap_or(2.0);
    let (input_cols, input_rows) = input.dim();
    let num_components: f64 = num_components.try_into().unwrap_or(input_cols as f64);
    let input = {
        if input_cols == 0 || input_rows == 0 {
            return Err(AARCError::PredictionDataSizeZero);
        }
        let v: Vec<f32> = input.into();
        trace!(
            "Input rows:{} cols:{} Total:{}",
            input_cols,
            input_rows,
            v.len()
        );
        let mut input = Array::from(v);
        input.reshape(input_rows, input_cols);
        input
    };
    let train = {
        let (_train_cols, train_rows) = train.dim();
        if train_rows == 0 || (train_rows != input_rows) {
            return Err(AARCError::PredictionDataSizeZero);
        }
        let v: Vec<f32> = train.into();
        trace!("Training rows:{} cols:1 Total:{}", train_rows, v.len());
        let mut train = Array::from(v);
        train.reshape(train_rows, 1);
        train
    };

    trace!(
        "LearningRate {} L1:{} L2: {} Crossfold:{} Epochs:{} ",
        learning_rate,
        l1_penalty,
        l2_penalty,
        cross_fold,
        epochs,
    );

    match rustlearn_if::train_factorization(
        &input,
        &train,
        num_components as usize,
        learning_rate,
        l2_penalty,
        l1_penalty,
        cross_fold as usize,
        epochs as usize,
    ) {
        Ok(v) => Ok(Variant::as_float_array(v, 2, 1)),
        Err(e) => Err(e),
    }
}

pub fn predict_factorization(id: Variant, predict: Variant) -> Result<Variant, AARCError> {
    let id: f64 = id.try_into()?;
    let (predict_cols, predict_rows) = predict.dim();
    if predict_rows == 0 || predict_cols == 0 {
        return Err(AARCError::PredictionDataSizeZero);
    }
    let predict = {
        let v: Vec<f32> = predict.into();
        trace!(
            "Predict rows:{} cols:{} Total:{}",
            predict_cols,
            predict_rows,
            v.len()
        );
        let mut predict = Array::from(v);
        predict.reshape(predict_rows, predict_cols);
        predict
    };
    match rustlearn_if::predict_factorization(id, predict) {
        Ok(v) => {
            let v = v
                .data()
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            trace!("Returning {:?}", v);
            let height = v.len();
            Ok(Variant::as_float_array(v, 1, height))
        }
        Err(e) => Err(e),
    }
}
