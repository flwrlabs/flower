use crate::typing as local;

pub trait Client {
    /// Return the current local model parameters
    fn get_parameters(&self) -> local::GetParametersRes;

    fn get_properties(&self, ins: local::GetPropertiesIns) -> local::GetPropertiesRes;

    /// Refine the provided weights using the locally held dataset
    ///
    /// The training instructions contain (global) model parameters
    /// received from the server and a dictionary of configuration
    /// values used to customize the local training process.
    fn fit(&self, ins: local::FitIns) -> local::FitRes;

    /// Evaluate the provided weights using the locally held dataset.
    ///
    /// The evaluation instructions contain (global) model parameters
    /// received from the server and a dictionary of configuration
    /// values used to customize the local evaluation process.
    fn evaluate(&self, ins: local::EvaluateIns) -> local::EvaluateRes;
}
