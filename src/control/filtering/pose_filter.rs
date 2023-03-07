use anyhow::{Context, Result};
use nalgebra::{
    vector, Complex, ComplexField, Isometry2, Matrix2, Matrix3, Matrix3x2, UnitComplex, Vector2,
    Vector3,
};
use serde::{Deserialize, Serialize};
use serialize_hierarchy::SerializeHierarchy;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, SerializeHierarchy)]

// this is used as an entry for localization data samples
pub struct ScoredPoseFilter {
    pub pose_filter: PoseFilter,  // the position state (mean and covariance)
    pub score: f32,  // ? what is score here (probability?)
}

impl ScoredPoseFilter {
    // isometry is measured information, mapped to a specific metric space
    pub fn from_isometry(pose: Isometry2<f32>, covariance: Matrix3<f32>, score: f32) -> Self {
        Self {
            // a pose filter is a mean state (x,y,theta) and covariance matrix
            pose_filter: PoseFilter::new(
                vector![
                    pose.translation.x,
                    pose.translation.y,
                    pose.rotation.angle()
                ],
                covariance,
            ),
            // ? score ?
            score,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, SerializeHierarchy)]
pub struct PoseFilter {
    mean: Vector3<f32>,
    covariance: Matrix3<f32>,
}

impl PoseFilter {
    pub fn new(initial_state: Vector3<f32>, inital_state_covariance: Matrix3<f32>) -> Self {
        Self {
            mean: initial_state,
            covariance: inital_state_covariance,
        }
    }

    // predict a new state
    pub fn predict<StatePredictionFunction>(
        &mut self,
        state_prediction_function: StatePredictionFunction,  // lambda function for prediction
        process_noise: Matrix3<f32>,  // process noise
    ) -> Result<()>
    where
        StatePredictionFunction: Fn(Vector3<f32>) -> Vector3<f32>,  // definition of the lambda function
    {

        // the following is Unscented Kalman Filter theory

        // we get some sigma points around the current location (returns error if not OK)
        let sigma_points = sample_sigma_points(self.mean, self.covariance)?;

        // predict new state for every sigma point
        let predicted_sigma_points: Vec<_> = sigma_points
            .iter()
            .copied()
            .map(state_prediction_function)  // apply the prediction function (e.g state.x + odometry.x)
            .collect();

        // get a mean from all the sigma points
        let state_mean = mean_from_3d_sigma_points(&predicted_sigma_points);

        // get a covariance from all the sigma points
        let state_covariance = covariance_from_3d_sigma_points(state_mean, &predicted_sigma_points);

        // actually update the mean of the PoseFilter object
        self.mean = state_mean;

        // actually update covariance, but make sure it's symmetric for kalman filter
        self.covariance = into_symmetric(state_covariance + process_noise);

        Ok(())
    }

    pub fn update_with_1d_translation_and_rotation<MeasurementPredictionFunction>(
        &mut self,
        measurement: Vector2<f32>,
        measurement_noise: Matrix2<f32>,
        measurement_prediction_function: MeasurementPredictionFunction,
    ) -> Result<()>
    where
        MeasurementPredictionFunction: Fn(Vector3<f32>) -> Vector2<f32>,
    {
        let sigma_points = sample_sigma_points(self.mean, self.covariance)?;
        let predicted_measurements: Vec<_> = sigma_points
            .iter()
            .copied()
            .map(measurement_prediction_function)
            .collect();
        let predicted_measurement_mean =
            mean_from_1d_translation_and_rotation_sigma_points(&predicted_measurements);
        let predicted_measurement_covariance =
            covariance_from_1d_translation_and_rotation_sigma_points(
                predicted_measurement_mean,
                &predicted_measurements,
            );

        let predicted_measurements_cross_covariance =
            cross_covariance_from_1d_translation_and_rotation_sigma_points(
                self.mean,
                &sigma_points,
                &predicted_measurement_mean,
                &predicted_measurements,
            );
        let kalman_gain = predicted_measurements_cross_covariance
            * (predicted_measurement_covariance + measurement_noise)
                .try_inverse()
                .context("Failed to invert measurement covariance matrix")?;

        let residuum = measurement - predicted_measurement_mean;
        self.mean += kalman_gain * residuum;
        let updated_state_covariance = self.covariance
            - kalman_gain * predicted_measurement_covariance * kalman_gain.transpose();
        self.covariance = into_symmetric(updated_state_covariance);

        Ok(())
    }

    // TODO: reduce code duplication
    pub fn update_with_2d_translation<MeasurementPredictionFunction>(
        &mut self,
        measurement: Vector2<f32>,
        measurement_noise: Matrix2<f32>,
        measurement_prediction_function: MeasurementPredictionFunction,
    ) -> Result<()>
    where
        MeasurementPredictionFunction: Fn(Vector3<f32>) -> Vector2<f32>,
    {
        let sigma_points = sample_sigma_points(self.mean, self.covariance)?;
        let predicted_measurements: Vec<_> = sigma_points
            .iter()
            .copied()
            .map(measurement_prediction_function)
            .collect();
        let predicted_measurement_mean =
            mean_from_2d_translation_sigma_points(&predicted_measurements);
        let predicted_measurement_covariance = covariance_from_2d_translation_sigma_points(
            predicted_measurement_mean,
            &predicted_measurements,
        );

        let predicted_measurements_cross_covariance =
            cross_covariance_from_2d_translation_sigma_points(
                self.mean,
                &sigma_points,
                &predicted_measurement_mean,
                &predicted_measurements,
            );
        let kalman_gain = predicted_measurements_cross_covariance
            * (predicted_measurement_covariance + measurement_noise)
                .try_inverse()
                .context("Failed to invert measurement covariance matrix")?;

        let residuum = measurement - predicted_measurement_mean;
        self.mean += kalman_gain * residuum;
        let updated_state_covariance = self.covariance
            - kalman_gain * predicted_measurement_covariance * kalman_gain.transpose();
        self.covariance = into_symmetric(updated_state_covariance);

        Ok(())
    }

    pub fn mean(&self) -> Vector3<f32> {
        self.mean
    }

    // rigid body transformation (shape and relative distance does not change)
    pub fn isometry(&self) -> Isometry2<f32> {
        Isometry2::new(vector![self.mean.x, self.mean.y], self.mean.z)
    }

    #[allow(dead_code)]
    pub fn covariance(&self) -> Matrix3<f32> {
        self.covariance
    }
}

fn into_symmetric(matrix: Matrix3<f32>) -> Matrix3<f32> {
    0.5 * (matrix + matrix.transpose())
}

fn sample_sigma_points(mean: Vector3<f32>, covariance: Matrix3<f32>) -> Result<[Vector3<f32>; 7]> {
    let covariance_cholesky = covariance.cholesky().with_context(|| {
        format!(
            "Failed to decompose covariance matrix via Cholesky decomposition. Matrix was: {}",
            covariance
        )
    })?;
    let covariance_square_root = covariance_cholesky.l();

    let sigma_points = [
        mean,
        mean + covariance_square_root.column(0),
        mean - covariance_square_root.column(0),
        mean + covariance_square_root.column(1),
        mean - covariance_square_root.column(1),
        mean + covariance_square_root.column(2),
        mean - covariance_square_root.column(2),
    ];
    Ok(sigma_points)
}

fn mean_from_3d_sigma_points(points: &[Vector3<f32>]) -> Vector3<f32> {
    let mut mean = Vector2::zeros();
    let mut mean_angle = Complex::new(0.0, 0.0);
    for point in points {
        mean += point.xy();
        mean_angle += Complex::new(point.z.cos(), point.z.sin());
    }
    mean *= 1.0 / 7.0;
    vector![mean.x, mean.y, mean_angle.argument()]
}

fn mean_from_1d_translation_and_rotation_sigma_points(points: &[Vector2<f32>]) -> Vector2<f32> {
    let mut mean_x = 0.0;
    let mut mean_angle = Complex::new(0.0, 0.0);
    for point in points {
        mean_x += point.x;
        mean_angle += Complex::new(point.y.cos(), point.y.sin());
    }
    mean_x *= 1.0 / 7.0;
    vector![mean_x, mean_angle.argument()]
}

fn mean_from_2d_translation_sigma_points(points: &[Vector2<f32>]) -> Vector2<f32> {
    let mut mean = Vector2::zeros();
    for point in points {
        mean += point;
    }
    mean *= 1.0 / 7.0;
    mean
}

fn covariance_from_3d_sigma_points(
    mean: Vector3<f32>,
    sigma_points: &[Vector3<f32>],
) -> Matrix3<f32> {
    sigma_points
        .iter()
        .map(|point| {
            vector![
                point.x - mean.x,
                point.y - mean.y,
                (UnitComplex::new(point.z) / UnitComplex::new(mean.z)).angle()
            ]
        })
        .map(|normalized_point| normalized_point * normalized_point.transpose())
        .sum::<Matrix3<f32>>()
        * (1.0 / 6.0)
}

fn covariance_from_1d_translation_and_rotation_sigma_points(
    mean: Vector2<f32>,
    sigma_points: &[Vector2<f32>],
) -> Matrix2<f32> {
    sigma_points
        .iter()
        .map(|point| {
            vector![
                point.x - mean.x,
                (UnitComplex::new(point.y) / UnitComplex::new(mean.y)).angle()
            ]
        })
        .map(|normalized_point| normalized_point * normalized_point.transpose())
        .sum::<Matrix2<f32>>()
        * (1.0 / 6.0)
}

fn covariance_from_2d_translation_sigma_points(
    mean: Vector2<f32>,
    sigma_points: &[Vector2<f32>],
) -> Matrix2<f32> {
    sigma_points
        .iter()
        .map(|point| point - mean)
        .map(|normalized_point| normalized_point * normalized_point.transpose())
        .sum::<Matrix2<f32>>()
        * (1.0 / 6.0)
}

fn cross_covariance_from_1d_translation_and_rotation_sigma_points(
    state_mean: Vector3<f32>,
    state_sigma_points: &[Vector3<f32>],
    &measurement_mean: &Vector2<f32>,
    measurement_sigma_points: &[Vector2<f32>],
) -> Matrix3x2<f32> {
    assert!(state_sigma_points.len() == measurement_sigma_points.len());
    state_sigma_points
        .iter()
        .zip(measurement_sigma_points.iter())
        .map(|(state, measurement)| {
            vector![
                state.x - state_mean.x,
                state.y - state_mean.y,
                (UnitComplex::new(state.z) / UnitComplex::new(state_mean.z)).angle()
            ] * vector![
                measurement.x - measurement_mean.x,
                (UnitComplex::new(measurement.y) / UnitComplex::new(measurement_mean.y)).angle()
            ]
            .transpose()
        })
        .sum::<Matrix3x2<f32>>()
        * (1.0 / 6.0)
}

fn cross_covariance_from_2d_translation_sigma_points(
    state_mean: Vector3<f32>,
    state_sigma_points: &[Vector3<f32>],
    &measurement_mean: &Vector2<f32>,
    measurement_sigma_points: &[Vector2<f32>],
) -> Matrix3x2<f32> {
    assert!(state_sigma_points.len() == measurement_sigma_points.len());
    state_sigma_points
        .iter()
        .zip(measurement_sigma_points.iter())
        .map(|(state, measurement)| {
            vector![
                state.x - state_mean.x,
                state.y - state_mean.y,
                (UnitComplex::new(state.z) / UnitComplex::new(state_mean.z)).angle()
            ] * (measurement - measurement_mean).transpose()
        })
        .sum::<Matrix3x2<f32>>()
        * (1.0 / 6.0)
}
