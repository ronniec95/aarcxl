use failure::Fail;
use std::borrow::Borrow;
use xladd::variant::{Variant, XLAddError};
use xladd::xlcall::LPXLOPER12;

#[derive(Debug, Fail)]
pub enum AARCError {
    #[fail(display = "F64 Conversion failure")]
    ExcelF64ConversionError,
    #[fail(display = "Bool Conversion failure")]
    ExcelBoolConversionError,
    #[fail(display = "Conversion failure")]
    ExcelStrConversionError,
    #[fail(display = "Model by id {} not found", _0)]
    ModelNotFoundError(u32),
    #[fail(display = "Logistic Regression Error  {} ", _0)]
    LogisticRegressionError(String),
    #[fail(display = "Prediction data size is zero, select more data")]
    PredictionDataSizeZero,
}

impl From<XLAddError> for AARCError {
    fn from(err: XLAddError) -> AARCError {
        match err {
            XLAddError::F64ConversionFailed => AARCError::ExcelF64ConversionError,
            XLAddError::BoolConversionFailed => AARCError::ExcelBoolConversionError,
            XLAddError::IntConversionFailed => AARCError::ExcelF64ConversionError,
            XLAddError::StringConversionFailed => AARCError::ExcelStrConversionError,
        }
    }
}

impl From<AARCError> for LPXLOPER12 {
    fn from(err: AARCError) -> LPXLOPER12 {
        LPXLOPER12::from(Variant::from(format!("{}", err).borrow()))
    }
}
