use crate::basic_stats::*;
use log::*;
use simplelog::*;
use std::fs::File;
use xladd::registrator::Reg;
use xladd::variant::Variant;
use xladd::xlcall::LPXLOPER12;

// Utility functions
#[no_mangle]
pub extern "stdcall" fn xuVersion() -> LPXLOPER12 {
    let result = Box::new(Variant::from("aarcxl: version 0.1.1"));
    if let Ok(_) = WriteLogger::init(
        LevelFilter::Trace,
        Config::default(),
        File::create(r"aarc.log").unwrap(),
    ) {
        info!("Logging started.");
    };
    Box::into_raw(result) as LPXLOPER12
}

// Normalise and scale
#[no_mangle]
pub extern "stdcall" fn aarc_clear_objects() -> LPXLOPER12 {
    clear_objects();
    LPXLOPER12::from(Variant::from("All objects cleared"))
}

// Normalise and scale
#[no_mangle]
pub extern "stdcall" fn aarc_normalize(
    array: LPXLOPER12,
    min: LPXLOPER12,
    max: LPXLOPER12,
    scale: LPXLOPER12,
) -> LPXLOPER12 {
    match normalize(
        Variant::from(array),
        Variant::from(min),
        Variant::from(max),
        Variant::from(scale),
    ) {
        Ok(v) => LPXLOPER12::from(v),
        _ => LPXLOPER12::from(Variant::from("Invalid")),
    }
}

#[no_mangle]
pub extern "stdcall" fn aarc_train_logistic_reg(
    input: LPXLOPER12,
    train: LPXLOPER12,
    learning_rate: LPXLOPER12,
    l1_penalty: LPXLOPER12,
    l2_penalty: LPXLOPER12,
    crossfold: LPXLOPER12,
    epochs: LPXLOPER12,
) -> LPXLOPER12 {
    match train_logistic_reg(
        Variant::from(input),
        Variant::from(train),
        Variant::from(learning_rate),
        Variant::from(l1_penalty),
        Variant::from(l2_penalty),
        Variant::from(crossfold),
        Variant::from(epochs),
    ) {
        Ok(v) => LPXLOPER12::from(v),
        _ => LPXLOPER12::from(Variant::from("Invalid")),
    }
}

//
#[no_mangle]
pub extern "stdcall" fn aarc_predict_logistic_reg(
    model_id: LPXLOPER12,
    predict: LPXLOPER12,
) -> LPXLOPER12 {
    match predict_logistic_reg(Variant::from(model_id), Variant::from(predict)) {
        Ok(v) => LPXLOPER12::from(v),
        Err(e) => LPXLOPER12::from(e),
    }
}

#[no_mangle]
pub extern "stdcall" fn aarc_train_random_forest(
    input: LPXLOPER12,
    train: LPXLOPER12,
    num_trees: LPXLOPER12,
    min_samples_fit: LPXLOPER12,
    max_features: LPXLOPER12,
    max_depth: LPXLOPER12,
    crossfold: LPXLOPER12,
    epochs: LPXLOPER12,
) -> LPXLOPER12 {
    match train_random_forest(
        Variant::from(input),
        Variant::from(train),
        Variant::from(num_trees),
        Variant::from(min_samples_fit),
        Variant::from(max_features),
        Variant::from(max_depth),
        Variant::from(crossfold),
        Variant::from(epochs),
    ) {
        Ok(v) => LPXLOPER12::from(v),
        Err(e) => LPXLOPER12::from(e),
    }
}

//
#[no_mangle]
pub extern "stdcall" fn aarc_predict_random_forest(
    model_id: LPXLOPER12,
    predict: LPXLOPER12,
) -> LPXLOPER12 {
    match predict_random_forest(Variant::from(model_id), Variant::from(predict)) {
        Ok(v) => LPXLOPER12::from(v),
        Err(e) => LPXLOPER12::from(e),
    }
}

#[no_mangle]
pub extern "stdcall" fn aarc_train_factorization_model(
    input: LPXLOPER12,
    train: LPXLOPER12,
    num_components: LPXLOPER12,
    learning_rate: LPXLOPER12,
    l1_penalty: LPXLOPER12,
    l2_penalty: LPXLOPER12,
    crossfold: LPXLOPER12,
    epochs: LPXLOPER12,
) -> LPXLOPER12 {
    match train_factorization(
        Variant::from(input),
        Variant::from(train),
        Variant::from(num_components),
        Variant::from(learning_rate),
        Variant::from(l1_penalty),
        Variant::from(l2_penalty),
        Variant::from(crossfold),
        Variant::from(epochs),
    ) {
        Ok(v) => LPXLOPER12::from(v),
        Err(e) => LPXLOPER12::from(e),
    }
}

//
#[no_mangle]
pub extern "stdcall" fn aarc_predict_factorization_model(
    model_id: LPXLOPER12,
    predict: LPXLOPER12,
) -> LPXLOPER12 {
    match predict_factorization(Variant::from(model_id), Variant::from(predict)) {
        Ok(v) => LPXLOPER12::from(v),
        Err(e) => LPXLOPER12::from(e),
    }
}

#[no_mangle]
pub extern "stdcall" fn xlAutoOpen() -> i32 {
    let r = Reg::new();
    r.add(
        "xuVersion",
        "Q$",
        "",
        "xladd-util",
        "Displays xladd-util version number as text.",
        &[],
    );
    r.add(
        "aarc_train_logistic_reg",
        "QQQQQQQQ",
        "input data, learning data, learning rate, l1 penalty, l2 penalty, epochs, crossfold",
        "xladd-util",
        "Learn using logistic regression",
        &[
            "Input data array",
            "Learning data",
            "Learning rate [0-1]",
            "L1 Penalty",
            "L2 Penalty",
            "Number of epochs to run (default 2)",
            "Crossfold (default 2)",
        ],
    );

    r.add(
        "aarc_predict_logistic_reg",
        "QQQ",
        "model id, training data",
        "xladd-util",
        "Predict using trained model",
        &["Model id from trained model", "Prediction data"],
    );

    r.add(
        "aarc_train_random_forest",
        "QQQQQQQQQ",
        "input data, learning data, num_trees,min_samples_fit,max_features,max_depth, epochs, crossfold",
        "xladd-util",
        "Learn using logistic regression",
        &[
            "Input data array",
            "Learning data",
            "num_trees",
            "min_samples_fit",
            "max_features",
            "max_depth",
            "Number of epochs to run (default 2)",
            "Crossfold (default 2)",
        ],
    );

    r.add(
        "aarc_predict_random_forest",
        "QQQ",
        "model id, training data",
        "xladd-util",
        "Predict using trained model",
        &["Model id from trained model", "Prediction data"],
    );

    r.add(
        "aarc_train_factorization_model",
        "QQQQQQQQQ",
        "input data, learning data, num components, learning rate, l1 penalty, l2 penalty, epochs, crossfold",
        "xladd-util",
        "Learn using logistic regression",
        &[
            "Input data array",
            "Learning data",
            "Number components",
            "Learning rate [0-1]",
            "L1 Penalty",
            "L2 Penalty",
            "Number of epochs to run (default 2)",
            "Crossfold (default 2)",
        ],
    );

    r.add(
        "aarc_predict_factorization_model",
        "QQQ",
        "model id, training data",
        "xladd-util",
        "Predict using trained model",
        &["Model id from trained model", "Prediction data"],
    );

    r.add(
        "aarc_normalize",
        "QQQQQ$",
        "Input data, min, max, type",
        "xladd-util",
        "Normalize and scale a value",
        &[
            "Input array",
            "Scale minimum",
            "Scale maximum",
            "Scale Type 0: MinMax 1: Zscore 2: Tanh",
        ],
    );

    r.add(
        "aarc_clear_objects",
        "Q$",
        "",
        "xladd-util",
        "Delete all the models",
        &[],
    );
    1
}
