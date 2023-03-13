use std::{f32::consts::FRAC_PI_2, mem::take};  // 0.5pi, take is taking ownership

use anyhow::{Context, Result};  // wraps error within more context
use approx::assert_relative_eq;  // crate for approximating equality
use module_derive::{module, require_some};  // those are macro functions to check if not None
use nalgebra::{
    distance, matrix, point, vector, Isometry2, Matrix, Matrix2, Matrix3, Point2, Rotation2,
    Vector2, Vector3,
};  // crate for linalg
use ordered_float::NotNan;  // wrapper around Float, guaranteed to not be NaN
use spl_network::{GamePhase, PlayerNumber, Team};
use types::{
    field_marks_from_field_dimensions, CorrespondencePoints, Direction, FieldDimensions, FieldMark,
    GameControllerState, InitialPose, Line, Line2, LineData, LocalizationUpdate, Players,
    PrimaryState, Side,
};  // custom types

// estimated position filters 
use crate::control::filtering::{PoseFilter, ScoredPoseFilter};

// ??
pub struct Localization {
    // vector of arbitrary fieldmarks (e.g. line, circle)
    field_marks: Vec<FieldMark>,  
    last_primary_state: PrimaryState,  // state given by gamecontroller (e.g. ready, set, penalized)
    hypotheses: Vec<ScoredPoseFilter>,  // all possible poses [src/control/filtering/pose_filter]
    hypotheses_when_entered_playing: Vec<ScoredPoseFilter>,  // ?? is this the initial hypotheses?
    is_penalized_with_motion_in_set: bool, // ?? boolean why matter
    was_picked_up_while_penalized_with_motion_in_set: bool, // ?? boolean why matter
}
// test-for-github
// parameters
//
#[module(control)]
#[input(path = primary_state, data_type = PrimaryState)]
#[input(path = game_controller_state, data_type = GameControllerState)]
#[input(path = has_ground_contact, data_type = bool)]
#[historic_input(path = current_odometry_to_last_odometry, data_type = Isometry2<f32>)]
#[perception_input(name = line_data_top, path = line_data, data_type = LineData, cycler = vision_top)]
#[perception_input(name = line_data_bottom, path = line_data, data_type = LineData, cycler = vision_bottom)]
#[persistent_state(path = robot_to_field, data_type = Isometry2<f32>)]
#[parameter(path = field_dimensions, data_type = FieldDimensions)]
#[parameter(path = control.localization.circle_measurement_noise, data_type = Vector2<f32>)]
#[parameter(path = control.localization.gradient_convergence_threshold, data_type = f32)]
#[parameter(path = control.localization.gradient_descent_step_size, data_type = f32)]
#[parameter(path = control.localization.hypothesis_prediction_score_reduction_factor, data_type = f32)]
#[parameter(path = control.localization.hypothesis_retain_factor, data_type = f32)]
#[parameter(path = control.localization.initial_hypothesis_covariance, data_type = Matrix3<f32>)]
#[parameter(path = control.localization.initial_hypothesis_score, data_type = f32)]
#[parameter(path = control.localization.initial_poses, data_type = Players<InitialPose>)]
#[parameter(path = control.localization.line_length_acceptance_factor, data_type = f32)]
#[parameter(path = control.localization.line_measurement_noise, data_type = Vector2<f32>)]
#[parameter(path = control.localization.maximum_amount_of_gradient_descent_iterations, data_type = usize)]
#[parameter(path = control.localization.maximum_amount_of_outer_iterations, data_type = usize)]
#[parameter(path = control.localization.minimum_fit_error, data_type = f32)]
#[parameter(path = control.localization.odometry_noise, data_type = Vector3<f32>)]
#[parameter(path = control.localization.use_line_measurements, data_type = bool)]
#[parameter(path = control.localization.good_matching_threshold, data_type = f32)]
#[parameter(path = control.localization.score_per_good_match, data_type = f32)]
#[parameter(path = control.localization.hypothesis_score_base_increase, data_type = f32)]
#[parameter(path = player_number, data_type = PlayerNumber)]
#[additional_output(path = localization.pose_hypotheses, data_type = Vec<ScoredPoseFilter>)]
#[additional_output(path = localization.correspondence_lines, data_type = Vec<Line2>)]
#[additional_output(path = localization.measured_lines_in_field, data_type = Vec<Line2>)]
#[additional_output(path = localization.updates, data_type = Vec<Vec<LocalizationUpdate>>)]
#[additional_output(path = localization.fit_errors, data_type = Vec<Vec<Vec<Vec<f32>>>>)]
#[main_output(name = robot_to_field, data_type = Isometry2<f32>)]
impl Localization {} // ??

impl Localization {
    // create a new localization instance
    // ? why even 
    fn new(context: NewContext) -> anyhow::Result<Self> {
        Ok(Self {
            // get the field and line marks and chain them
            field_marks: field_marks_from_field_dimensions(context.field_dimensions)
                .into_iter()
                .chain(goal_support_structure_line_marks_from_field_dimensions(
                    context.field_dimensions,
                ))
                .collect(),

            last_primary_state: PrimaryState::Unstiff,

            // ? what are the differences between these hypotheses
            hypotheses: vec![],
            hypotheses_when_entered_playing: vec![],

            // for a new localization instance default boolean false
            is_penalized_with_motion_in_set: false,
            was_picked_up_while_penalized_with_motion_in_set: false,
        })
    }

    // !! ??
    // ? 1 wat is cycle context
    // ? 2 wat is primary state - state gegeven door game controller
    // ? 3 hoe werkt game controller state - game state, phase, kicking team etc
    // ? 4 verschil hypotheses en when entered playing - 
    fn cycle(&mut self, mut context: CycleContext) -> Result<MainOutputs> {
        /*
        This is the Localization cycle, which handles all parts of the localization algorithms
        run every tick.

        Steps:
            1. Get information from game controller
            2. ---- Match statement ----
            3. Gather measured lines from cameras, for each:
                a. Gather odometry data
                b. For every hypothesis
                    I. Prediction phase using odometry
                    II. Observation phase using fieldmark-correspondences from measured lines
            4. Update best hypothesis and the hypotheses that are kept
            5. 
         */
        // get several values from primary state and game controller state
        let primary_state = *require_some!(context.primary_state);

        let penalty = context
            .game_controller_state
            .map(|game_controller_state| game_controller_state.penalties[*context.player_number])
            .flatten();
        let game_phase = context
            .game_controller_state
            .map(|game_controller_state| game_controller_state.game_phase);

        // 
        let has_ground_contact = *require_some!(context.has_ground_contact);
        

        // ? how does match work here
        // ? what are different match cases - generally
        // while answering this you can ask new question about specific matches
        

        // !! THIS LOOKS LIKE SOMETHING WITH FSM
        // ? ? something like changing the current primary state

        // match last primary state, primary state and game phase
        match (self.last_primary_state, primary_state, game_phase) {
            // if last primary is initial, primary is ready (looks like a starting case) create initial pose and hypotheses
            (PrimaryState::Initial, PrimaryState::Ready, _) => {

                // ? what is an initial pose
                // initialize initial pose and get values from context
                let initial_pose = generate_initial_pose(

                    // ? which context is used here
                    &context.initial_poses[*context.player_number],
                    context.field_dimensions,
                );
                // initialize initial hypotheses
                self.hypotheses = vec![ScoredPoseFilter::from_isometry(
                    initial_pose,
                    *context.initial_hypothesis_covariance,
                    *context.initial_hypothesis_score,
                )];
                self.hypotheses_when_entered_playing = self.hypotheses.clone();
            }
            // if last primary is set, primary is playing and game phase is penalty shoot whith Hulks kicking
            // create hypotheses with striker pose
            (
                PrimaryState::Set,
                PrimaryState::Playing,
                Some(GamePhase::PenaltyShootout {
                    kicking_team: Team::Hulks,
                }),
            ) => {
                // create striker pose
                // ?? this looks like te location of the penalty taker 
                let penalty_shoot_out_striker_pose = Isometry2::translation(
                    -context.field_dimensions.penalty_area_length
                        + (context.field_dimensions.length / 2.0),
                    0.0,
                );
                // create hypotheses with same (initial) values but different (striker) pose
                self.hypotheses = vec![ScoredPoseFilter::from_isometry(
                    penalty_shoot_out_striker_pose,
                    *context.initial_hypothesis_covariance,
                    *context.initial_hypothesis_score,
                )];
                self.hypotheses_when_entered_playing = self.hypotheses.clone();
            }
            // if last primary is set, primary is playing and game phase is penalty shoot out but for opponent
            // create hypotheses with keeper pose
            (
                PrimaryState::Set,
                PrimaryState::Playing,
                Some(GamePhase::PenaltyShootout {
                    kicking_team: Team::Opponent,
                }),
            ) => {
                // create pose of kepper
                // ?? location
                let penalty_shoot_out_keeper_pose =
                    Isometry2::translation(-context.field_dimensions.length / 2.0, 0.0);
                // create hypotheses with same (initial) values but different (keeper) pose
                self.hypotheses = vec![ScoredPoseFilter::from_isometry(
                    penalty_shoot_out_keeper_pose,
                    *context.initial_hypothesis_covariance,
                    *context.initial_hypothesis_score,
                )];
                self.hypotheses_when_entered_playing = self.hypotheses.clone();
            }
            // if last primary is set and primary is playing, use current hypotheses and only clone it
            (PrimaryState::Set, PrimaryState::Playing, _) => {
                self.hypotheses_when_entered_playing = self.hypotheses.clone();
            }
            (PrimaryState::Playing, PrimaryState::Penalized, _) => {
                match penalty {
                    Some(spl_network::Penalty::IllegalMotionInSet { remaining: _ }) => {


                        //?? this is a boolean in the loc struct, why true?
                        self.is_penalized_with_motion_in_set = true;
                    }
                    Some(_) => {}
                    None => {}
                };
            }
            (PrimaryState::Penalized, _, _) if primary_state != PrimaryState::Penalized => {

                // ?? why does this boolean matter here and what does it do
                if self.is_penalized_with_motion_in_set {

                    // ?? why does this other boolean matter here and what does it do?
                    if self.was_picked_up_while_penalized_with_motion_in_set {
                        self.hypotheses = take(&mut self.hypotheses_when_entered_playing);

                        let penalized_poses = generate_penalized_poses(context.field_dimensions);
                        self.hypotheses_when_entered_playing = penalized_poses
                            .into_iter()
                            .map(|pose| {
                                ScoredPoseFilter::from_isometry(
                                    pose,
                                    *context.initial_hypothesis_covariance,
                                    *context.initial_hypothesis_score,
                                )
                            })
                            .collect();
                    }

                    // ! there are those booleans again, back to default false
                    self.is_penalized_with_motion_in_set = false;
                    self.was_picked_up_while_penalized_with_motion_in_set = false;
                } else {
                    let penalized_poses = generate_penalized_poses(context.field_dimensions);
                    self.hypotheses = penalized_poses
                        .into_iter()
                        .map(|pose| {
                            ScoredPoseFilter::from_isometry(
                                pose,
                                *context.initial_hypothesis_covariance,
                                *context.initial_hypothesis_score,
                            )
                        })
                        .collect();
                    self.hypotheses_when_entered_playing = self.hypotheses.clone();
                }
            }
            (PrimaryState::Unstiff, _, _) => {
                let penalized_poses = generate_penalized_poses(context.field_dimensions);
                self.hypotheses = penalized_poses
                    .into_iter()
                    .map(|pose| {
                        ScoredPoseFilter::from_isometry(
                            pose,
                            *context.initial_hypothesis_covariance,
                            *context.initial_hypothesis_score,
                        )
                    })
                    .collect();
                self.hypotheses_when_entered_playing = self.hypotheses.clone();
            }
            _ => {}
        }



        // end of match

        
        // current primary state will be last
        self.last_primary_state = primary_state;

        // ! booleans back at it again
        // if robot is penalized and not on ground it has been picked up
        if self.is_penalized_with_motion_in_set && !has_ground_contact {
            self.was_picked_up_while_penalized_with_motion_in_set = true;
        }
        
        // if state is ready, set or playing, we can start updating our localization
        if primary_state == PrimaryState::Ready
            || primary_state == PrimaryState::Set
            || primary_state == PrimaryState::Playing
        {   
            
            /*
            DataType below is: Vec<Vec<Vec<Vec<f32>>>>
            Basically for each measurement we have information for all hypotheses.
            
            1. add all hypotheses info for each measurement
                - var: fit_errors_per_measurement
                - how: fit_errors_per_measurement.push(fit_errors)
                - type: Vec<Vec<Vec<Vec<f32>>>>

                2. add all outer iterations for each hypothesis
                    - var: fit_errors_per_hypothesis
                    - how: fit_errors_per_hypothesis.push(fit_errors)
                    - type: Vec<Vec<Vec<f32>>>
                
                    3. add GradDesc fit errors for each outer iteration
                        - var: fit_errors
                        - how: fit_errors.push(fit_errors_per_iteration)
                        - type: Vec<Vec<f32>>
                
                        4. add fit errors for each GradDesc iteration
                        - var: fit_errors_per_iteration
                        - how: fit_errors_per_iteration.push(error)
                        - type: Vec<f32>
                
                            5. get an error for a specific GradDesc iteration
                                - var: error
                                - how: get_fit_error
                                - type: <f32>
            
            
            
             */
            let mut fit_errors_per_measurement: Vec<Vec<Vec<Vec<f32>>>> = vec![];

            // from the CycleContext get the measured lines
            context
                .measured_lines_in_field // AdditionalOutput<Vec<Line<...>>>
                .fill_on_subscription(Vec::new);
            context.correspondence_lines.fill_on_subscription(Vec::new);

            // ?? dont even know what this means
            // from CycleContext get the updates and fill them if they're subscribed
            context
                .updates // AdditionalOutput<Vec<Vec<...>>>
                .fill_on_subscription(|| vec![vec![]; self.hypotheses.len()]);
            
            
            // [src/vision/cycler.rs:pub fn start] for camera
            // [crates/types/src/line_data.rs]
            // LineData struct contains
            // - lines_in_robot: Vec<Line2>
            // - used_vertical_filtered_segments: HashSet<Point2<u16>>

            // zip line data from top and bottom together
            let line_datas = context    // CycleContext
                // perceptionDataType containing persistent and temporary timestamped LineData
                // [src/framework/perception_database.rs, pub struct PerceptionDataType]
                .line_data_top
                // get persistent data from top camera (vision top), contains: (timestamp, LineData)
                .persistent
                .iter()
                .zip(context.line_data_bottom.persistent.iter());

            // iterate through LineData in top and bottom camera
            for (
                (line_data_top_timestamp, line_data_top),  // &SystemTime, Vec<&Option<LineData>>
                (line_data_bottom_timestamp, line_data_bottom),  // &SystemTime, Vec<&Option<LineData>>
            ) in line_datas

            
            
            {
                // assert that top and bottom are synced -> otherwise panic 
                assert_eq!(line_data_top_timestamp, line_data_bottom_timestamp);

                /*
                [src/control/modules/odometry.rs]
                let current_odometry_to_last_odometry = Isometry2::from_parts(
                    Translation2::from(offset_to_last_position),
                    orientation_offset,
                );

                1. It uses offset_to_last_position which is calculated using:
                fn calculate_offset_to_last_position()

                

                2. It uses orientation_offset which is calculated using 
                last_rotation.rotation_to(&robot_orientation)\

                3. Those 2 offsets (basically difference through translation and rotation)
                are combined in a Isometry matrix
                
                */

                // get odometry values from the specific timestamp (could be invalid)
                let current_odometry_to_last_odometry = context

                    // this looks like an array of odometry values through time
                    .current_odometry_to_last_odometry

                    // getting the specific odometry value of a timestamp
                    .get(*line_data_top_timestamp);
                
                /*
                DataType below is: Vec<Vec<Vec<f32>>>

                1. add all outer iterations for each hypothesis
                    - var: fit_errors_per_hypothesis
                    - how: fit_errors_per_hypothesis.push(fit_errors)
                    - type: Vec<Vec<Vec<f32>>>
                
                    2. add GradDesc fit errors for each outer iteration
                        - var: fit_errors
                        - how: fit_errors.push(fit_errors_per_iteration)
                        - type: Vec<Vec<f32>>
                
                        3. add fit errors for each GradDesc iteration
                        - var: fit_errors_per_iteration
                        - how: fit_errors_per_iteration.push(error)
                        - type: Vec<f32>
                
                            4. get an error for a specific GradDesc iteration
                                - var: error
                                - how: get_fit_error
                                - type: <f32>

               

                BTW: This gets pushed into: fit_errors_per_measurement
                 */

                // store all outer iteration info for each hypothesis
                let mut fit_errors_per_hypothesis: Vec<Vec<Vec<f32>>> = vec![];

                // for every hypothesis (=possible robot location)
                // reminder: [src/control/filtering/pose_filter]
                for (hypothesis_index, scored_filter) in self.hypotheses.iter_mut().enumerate() {
                    
                    // PROCESS PREDICTION PHASE
                    // if the odometry values were valid make a prediction, otherwise not
                    if let Some(current_odometry_to_last_odometry) =
                        current_odometry_to_last_odometry
                    {   
                        // use the general predict function to get process noise data
                        // which internally calls PoseFilter.predict to update the pose filter state
                        predict(
                            &mut scored_filter.pose_filter,  // ScoredPoseFilter.PoseFilter
                            current_odometry_to_last_odometry,
                            context.odometry_noise,
                        )
                        .context("Failed to predict pose filter")?;

                        // ? what does this score mean?
                        // ? is it like the likelihood of the hypothesis?
                        scored_filter.score *=
                            *context.hypothesis_prediction_score_reduction_factor;
                    }


                    // OBSERVATION UPDATE PHASE
                    // if line measurements can be use, we use them to
                    // ? to what
                    if *context.use_line_measurements {

                        // get the rigid body transformation to field
                        let robot_to_field = scored_filter.pose_filter.isometry();

                        // get all measured lines in the field from top camera
                        let current_measured_lines_in_field: Vec<_> = line_data_top
                            .iter()

                            // combine the top lines with measured bottom lines
                            .chain(line_data_bottom.iter())

                            // functional programming ask Pete, basically removing Nones
                            .filter_map(|&data| data.as_ref())

                            // ? what is the transformation that is applied here ? 
                            .flat_map(|line_data| {
                                line_data
                                    .lines_in_robot
                                    .iter()

                                    // ? mappinggg????
                                    .map(|&measured_line_in_robot| {
                                        robot_to_field * measured_line_in_robot
                                    })
                            })
                            .collect();

                        // TODO
                        context.measured_lines_in_field.mutate_on_subscription(
                            |measured_lines_in_field| {
                                if let Some(measured_lines_in_field) = measured_lines_in_field {
                                    measured_lines_in_field
                                        .extend(current_measured_lines_in_field.iter());
                                }
                            },
                        );

                        // if there are no measured lines, we can't do anything with them
                        // so get to next hypothesis
                        if current_measured_lines_in_field.is_empty() {
                            continue;
                        }
                        
                        // create tuple for a field mark with avg fit error and all fit errors
                        // ? fit on what?
                        // ? what error
                        let (field_mark_correspondences, fit_error, fit_errors) =
                            get_fitted_field_mark_correspondence(
                                &current_measured_lines_in_field,
                                &self.field_marks,
                                &context,
                                context.fit_errors.is_subscribed(),
                            );

                        // TODO
                        context.correspondence_lines.mutate_on_subscription(
                            |correspondence_lines| {
                                let next_correspondence_lines = field_mark_correspondences
                                    .iter()
                                    .flat_map(|field_mark_correspondence| {
                                        let correspondence_points_0 =
                                            field_mark_correspondence.correspondence_points.0;
                                        let correspondence_points_1 =
                                            field_mark_correspondence.correspondence_points.1;
                                        [
                                            Line(
                                                correspondence_points_0.measured,
                                                correspondence_points_0.reference,
                                            ),
                                            Line(
                                                correspondence_points_1.measured,
                                                correspondence_points_1.reference,
                                            ),
                                        ]
                                    });
                                if let Some(correspondence_lines) = correspondence_lines {
                                    correspondence_lines.extend(next_correspondence_lines);
                                }
                            },
                        );
                        

                        // look into:
                        // [crates/framework/additional_output.rs]
                        // checks if fit_error is subscribed
                        // ? but where is this boolean changed?
                        if context.fit_errors.is_subscribed() {
                            fit_errors_per_hypothesis.push(fit_errors);
                        }

                        
                        
                        /*
                        Extra information about the code below:

                        clamping -> Dutch: geklemd

                        The term "clamped" refers to the fact that the value of clamped_fit_error is limited or 
                        "clamped" to be within a certain range, 
                        specifically the range between *context.minimum_fit_error and 
                        the original value of fit_error. 
                        
                        Source: ChatGPT

                        Used to compute uncertainty_weight:

                        let uncertainty_weight = clamped_fit_error
                                * number_of_measurements_weight
                                * line_length_weight
                                * line_distance_to_robot;
                        */

                        // get the max between fit error and minimum fit error
                        let clamped_fit_error = fit_error.max(*context.minimum_fit_error);


                        /*
                        used to compute uncertainty_weight:

                        let uncertainty_weight = clamped_fit_error
                                * number_of_measurements_weight
                                * line_length_weight
                                * line_distance_to_robot;
                        */
                        let number_of_measurements_weight =
                            1.0 / field_mark_correspondences.len() as f32;
                        
                        
                        // for every hypothesis we look at every field mark correspondence
                        for field_mark_correspondence in field_mark_correspondences {

                            // we create an update
                            // ? is this a update based on the fieldmark?
                            // ? why do they differ?
                            let update = match field_mark_correspondence.field_mark {
                                // either the fieldmark is a line with a line update
                                FieldMark::Line { .. } => get_translation_and_rotation_measurement(
                                    robot_to_field,
                                    field_mark_correspondence,
                                ),

                                // either the fieldmark is a circle with a circle update
                                FieldMark::Circle { .. } => get_2d_translation_measurement(
                                    robot_to_field,
                                    field_mark_correspondence,
                                ),
                            };

                            // get the length of the measured line
                            // ! seems weird if it is not a line
                            let line_length =
                                field_mark_correspondence.measured_line_in_field.length();
                            
                            
                            /*
                            used to compute uncertainty_weight:

                            let uncertainty_weight = clamped_fit_error
                                    * number_of_measurements_weight
                                    * line_length_weight
                                    * line_distance_to_robot;
                            */

                            // when the length is 0, the weight is 1
                            let line_length_weight = if line_length == 0.0 {
                                1.0
                            } else {
                                1.0 / line_length
                            };

                            // get the center and distance from measured circle
                            // ! seems weird if it is not a circle

                            let line_center_point =
                                field_mark_correspondence.measured_line_in_field.center();


                            /*
                            used to compute uncertainty_weight:

                            let uncertainty_weight = clamped_fit_error
                                    * number_of_measurements_weight
                                    * line_length_weight
                                    * line_distance_to_robot;
                            */
                            let line_distance_to_robot = distance(
                                &line_center_point,
                                &Point2::from(robot_to_field.translation.vector),
                            );

                            // TODO unclear
                            context.updates.mutate_on_subscription(|updates| {
                                if let Some(updates) = updates {
                                    updates[hypothesis_index].push({
                                        let robot_to_field =
                                            match field_mark_correspondence.field_mark {
                                                FieldMark::Line { line: _, direction } => {
                                                    match direction {
                                                        Direction::PositiveX => Isometry2::new(
                                                            vector![
                                                                robot_to_field.translation.x,
                                                                update.x
                                                            ],
                                                            update.y,
                                                        ),
                                                        Direction::PositiveY => Isometry2::new(
                                                            vector![
                                                                update.x,
                                                                robot_to_field.translation.y
                                                            ],
                                                            update.y,
                                                        ),
                                                    }
                                                }
                                                FieldMark::Circle { .. } => Isometry2::new(
                                                    update,
                                                    robot_to_field.rotation.angle(),
                                                ),
                                            };
                                        LocalizationUpdate {
                                            robot_to_field,
                                            line_center_point,
                                            fit_error: clamped_fit_error,
                                            number_of_measurements_weight,
                                            line_distance_to_robot,
                                            line_length_weight,
                                        }
                                    });
                                }
                            });

                            // create the weight used for uncertainty
                            // ? why is this based on these variables ?
                            let uncertainty_weight = clamped_fit_error
                                * number_of_measurements_weight
                                * line_length_weight
                                * line_distance_to_robot;

                            
                            match field_mark_correspondence.field_mark {
                                // if the fieldmark is a line
                                FieldMark::Line { line: _, direction } => scored_filter
                                    .pose_filter
                                    .update_with_1d_translation_and_rotation(
                                        update,  // measurement
                                        Matrix::from_diagonal(context.line_measurement_noise)  // measurement_noise
                                            * uncertainty_weight,
                                        // a line has a direction which we can define

                                        // this is a measurement_prediction_function
                                        |state| match direction {
                                            Direction::PositiveX => {
                                                vector![state.y, state.z]
                                            }
                                            Direction::PositiveY => {
                                                vector![state.x, state.z]
                                            }
                                        },
                                    )
                                    .context("Failed to update pose filter")?,
                                
                                // if the fieldmark is a circle
                                FieldMark::Circle { .. } => scored_filter
                                    .pose_filter
                                    .update_with_2d_translation(
                                        update,
                                        Matrix::from_diagonal(context.circle_measurement_noise)
                                            * uncertainty_weight,
                                        |state| vector![state.x, state.y],
                                    )
                                    .context("Failed to update pose filter")?,
                            }

                            // check if the summed fit error is smaller than a good threshold
                            if field_mark_correspondence.fit_error_sum()
                                < *context.good_matching_threshold
                            {
                                // update the score with a positive score since it is a good match
                                scored_filter.score += *context.score_per_good_match;
                            }
                        }
                    }


                    // update the score of the hypothesis with the base increase
                    scored_filter.score += *context.hypothesis_score_base_increase;
                }

                // NOTE: This is the end of the hypothesis forloop

                if context.fit_errors.is_subscribed() {
                    fit_errors_per_measurement.push(fit_errors_per_hypothesis);
                }
            }

            // NOTE: This is the end of the LineData forloop

            // from all hypotheses we get the hypothesis with highest score
            // ! NOTE: score therefore seems to be related to the amount of matches with fieldmarks
            let best_hypothesis = self
                .get_best_hypothesis()
                .expect("Expected at least one hypothesis");

            // get the score of the best hypothesis
            let best_score = best_hypothesis.score;

            // get the rigid body transformation
            // ! ? is this the rotation to go from internal robot frame to field

            let robot_to_field = best_hypothesis.pose_filter.isometry();

            // ? what is a retain factor and why use it?
            // it seems like only hypothesis are kept which are at least in the neighbourhood of the best score
            self.hypotheses
                .retain(|filter| filter.score >= *context.hypothesis_retain_factor * best_score);

            // TODO unclear
            context
                .pose_hypotheses
                .fill_on_subscription(|| self.hypotheses.clone());
            context
                .fit_errors
                .fill_on_subscription(|| fit_errors_per_measurement);

            // return the robot_to_field value
            // ? what is this and why assign it again
            *context.robot_to_field = robot_to_field;
            return Ok(MainOutputs {
                robot_to_field: Some(robot_to_field),
            });
        }


        // if state is NOT ready, set or playing do nothing (?)
        Ok(MainOutputs {
            robot_to_field: None,
        })
    }
    
    // get the hypothesis with highest score from all scored hypotheses
    fn get_best_hypothesis(&self) -> Option<&ScoredPoseFilter> {
        self.hypotheses
            .iter()
            .max_by_key(|scored_filter| NotNan::new(scored_filter.score).unwrap())
    }
}

fn goal_support_structure_line_marks_from_field_dimensions(
    field_dimensions: &FieldDimensions,
) -> Vec<FieldMark> {
    let goal_width = field_dimensions.goal_inner_width + field_dimensions.goal_post_diameter;
    let goal_depth = field_dimensions.goal_depth;
    vec![
        FieldMark::Line {
            line: Line(
                point![
                    -field_dimensions.length / 2.0 - goal_depth,
                    -goal_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 - goal_depth,
                    goal_width / 2.0
                ],
            ),
            direction: Direction::PositiveY,
        },
        FieldMark::Line {
            line: Line(
                point![
                    -field_dimensions.length / 2.0 - goal_depth,
                    -goal_width / 2.0
                ],
                point![-field_dimensions.length / 2.0, -goal_width / 2.0],
            ),
            direction: Direction::PositiveX,
        },
        FieldMark::Line {
            line: Line(
                point![
                    -field_dimensions.length / 2.0 - goal_depth,
                    goal_width / 2.0
                ],
                point![-field_dimensions.length / 2.0, goal_width / 2.0],
            ),
            direction: Direction::PositiveX,
        },
        FieldMark::Line {
            line: Line(
                point![
                    field_dimensions.length / 2.0 + goal_depth,
                    -goal_width / 2.0
                ],
                point![field_dimensions.length / 2.0 + goal_depth, goal_width / 2.0],
            ),
            direction: Direction::PositiveY,
        },
        FieldMark::Line {
            line: Line(
                point![field_dimensions.length / 2.0, -goal_width / 2.0],
                point![
                    field_dimensions.length / 2.0 + goal_depth,
                    -goal_width / 2.0
                ],
            ),
            direction: Direction::PositiveX,
        },
        FieldMark::Line {
            line: Line(
                point![field_dimensions.length / 2.0, goal_width / 2.0],
                point![field_dimensions.length / 2.0 + goal_depth, goal_width / 2.0],
            ),
            direction: Direction::PositiveX,
        },
    ]
}

#[derive(Clone, Copy, Debug)]
struct FieldMarkCorrespondence {
    // Line struct with dimension set to 2 representing measured line
    measured_line_in_field: Line2,
    // Field mark that corresponds to measured line
    field_mark: FieldMark,
    // first tuple is correspondence points from line, second is from field mark
    // both have two (x, y) Point2 objects, measurement and reference respectively
    // why does line have reference and why does field mark have measurement?
    correspondence_points: (CorrespondencePoints, CorrespondencePoints),
}

impl FieldMarkCorrespondence {
    // sums errors for both line and field mark after normalizing
    fn fit_error_sum(&self) -> f32 {
        (self.correspondence_points.0.measured - self.correspondence_points.0.reference).norm()
            + (self.correspondence_points.1.measured - self.correspondence_points.1.reference)
                .norm()
    }
}


// predict a PoseFilter State based on current state and odometry
fn predict(
    filter: &mut PoseFilter,
    current_odometry_to_last_odometry: &Isometry2<f32>,
    odometry_noise: &Vector3<f32>,
) -> Result<()> {

    let current_orientation_angle = filter.mean().z;
    // rotate odometry noise from robot frame to field frame
    let rotated_noise = Rotation2::new(current_orientation_angle) * odometry_noise.xy();

    // create diagonal matrix with the diagonal equal to the vector with noise x,y,z
    // diagonality is a principle used for independence, efficiency and interpretability
    let process_noise = Matrix::from_diagonal(&vector![
        rotated_noise.x.abs(),
        rotated_noise.y.abs(),
        odometry_noise.z
    ]);
    
    // PoseFilter.predict [src/control/filtering/pose_filter.rs]
    filter.predict(
        // this is a lambda function that the predict function expects as state_prediction_function
        |state| {
            // rotate odometry from robot frame to field frame
            let robot_odometry =
                Rotation2::new(state.z) * current_odometry_to_last_odometry.translation.vector;

            // create a vector with updated states based on odometry
            vector![
                state.x + robot_odometry.x,
                state.y + robot_odometry.y,
                state.z + current_odometry_to_last_odometry.rotation.angle()
            ]
        },

        // this is the process noise the predict function expects
        process_noise,
    )
}

fn get_fitted_field_mark_correspondence(
    measured_lines_in_field: &[Line2],
    field_marks: &[FieldMark],
    context: &CycleContext,
    fit_errors_is_subscribed: bool,
) -> (Vec<FieldMarkCorrespondence>, f32, Vec<Vec<f32>>) {
    // create empty vector using vec macro for storing fit errors
    let mut fit_errors = vec![];
    // create isometry (translation, rotation, reflection etc) matrix, default is identity 
    let mut correction = Isometry2::identity();
    // context provides max number of iterations
    // probably based on 


    // ? outer iterations?
    for _ in 0..*context.maximum_amount_of_outer_iterations {
        // gets correspondence points
        let correspondence_points = get_correspondence_points(get_field_mark_correspondence(
            measured_lines_in_field,
            correction,
            field_marks,
            *context.line_length_acceptance_factor,
        ));

        // gets weight matrices from correspondence points
        let weight_matrices: Vec<_> = correspondence_points
            .iter()

            // manipulate the correspondence points to a weight matrix
            .map(|correspondence_points| {

                // first normalize by correcting the measured points by factor and minus reference
                let normal =
                    (correction * correspondence_points.measured) - correspondence_points.reference;

                // the measured point must be nonzero
                if normal.norm() > 0.0 {
                    // normalize the matrix
                    let normal_versor = normal.normalize();

                    // ? return a symmetric matrix or something?
                    normal_versor * normal_versor.transpose()
                } else {

                    Matrix2::zeros()
                }
            })
            .collect();
        
        // store fit error for each gradient descent iteration
        let mut fit_errors_per_iteration = vec![];


        // gradient descent iterations
        for _ in 0..*context.maximum_amount_of_gradient_descent_iterations {

            // compute translation gradient
            let translation_gradient: Vector2<f32> = correspondence_points
                .iter()
                .zip(weight_matrices.iter())

                // compute from correspondence points
                .map(|(correspondence_points, weight_matrix)| {
                    2.0 * weight_matrix
                        * ((correction * correspondence_points.measured)
                            - correspondence_points.reference)
                })
                .sum::<Vector2<f32>>()
                / correspondence_points.len() as f32;

            // compute rotation
            let rotation = correction.rotation.angle();

            // compute rotation derivative
            let rotation_derivative =
                matrix![-rotation.sin(), -rotation.cos(); rotation.cos(), -rotation.sin()];

            // compute rotation gradient
            let rotation_gradient: f32 = correspondence_points
                .iter()
                .zip(weight_matrices.iter())

                // compute from correspondence points
                .map(|(correspondence_points, weight_matrix)| {
                    (2.0 * correspondence_points.measured.coords.transpose()
                        * rotation_derivative.transpose()
                        * weight_matrix
                        * ((correction * correspondence_points.measured)
                            - correspondence_points.reference))
                        .x
                })
                .sum::<f32>()
                / correspondence_points.len() as f32;

            // compute a new correction matrix
            correction = Isometry2::new(
                correction.translation.vector
                    - *context.gradient_descent_step_size * translation_gradient,
                rotation - *context.gradient_descent_step_size * rotation_gradient,
            );

            // for current GradDesc iteration compute and add the fit error
            if fit_errors_is_subscribed {
                let error = get_fit_error(&correspondence_points, &weight_matrices, correction);
                fit_errors_per_iteration.push(error);
            }
            let gradient_norm = vector![
                translation_gradient.x,
                translation_gradient.y,
                rotation_gradient
            ]
            .norm();
            if gradient_norm < *context.gradient_convergence_threshold {
                break;
            }
        }

        // for current outer iteration add all GradDesc fit errors
        if fit_errors_is_subscribed {
            fit_errors.push(fit_errors_per_iteration);
        }
    }

    let field_mark_correspondences = get_field_mark_correspondence(
        measured_lines_in_field,
        correction,
        field_marks,
        *context.line_length_acceptance_factor,
    );

    let correspondence_points = get_correspondence_points(field_mark_correspondences.clone());
    let weight_matrices: Vec<_> = correspondence_points
        .iter()
        .map(|correspondence_points| {
            let normal =
                (correction * correspondence_points.measured) - correspondence_points.reference;
            if normal.norm() > 0.0 {
                let normal_versor = normal.normalize();
                normal_versor * normal_versor.transpose()
            } else {
                Matrix2::zeros()
            }
        })
        .collect();

    // TODO this seems like the average fit error
    let fit_error = get_fit_error(&correspondence_points, &weight_matrices, correction);
    
    // TODO field mark, avg fit error, all fit errors?
    (field_mark_correspondences, fit_error, fit_errors)
}

fn get_fit_error(
    correspondence_points: &[CorrespondencePoints],
    weight_matrices: &[Matrix2<f32>],
    correction: Isometry2<f32>,
) -> f32 {
    correspondence_points
        .iter()
        .zip(weight_matrices.iter())
        .map(|(correspondence_points, weight_matrix)| {
            ((correction * correspondence_points.measured - correspondence_points.reference)
                .transpose()
                * weight_matrix
                * (correction * correspondence_points.measured - correspondence_points.reference))
                .x
        })
        .sum::<f32>()
        / correspondence_points.len() as f32
}

fn get_field_mark_correspondence(
    measured_lines_in_field: &[Line2],
    correction: Isometry2<f32>,
    field_marks: &[FieldMark],
    line_length_acceptance_factor: f32,
) -> Vec<FieldMarkCorrespondence> {
    measured_lines_in_field
        .iter()
        .filter_map(|&measured_line_in_field| {
            let (correspondences, _weight, field_mark, transformed_line) = field_marks
                .iter()
                .filter_map(|field_mark| {
                    let transformed_line = correction * measured_line_in_field;
                    let field_mark_length = match field_mark {
                        FieldMark::Line { line, direction: _ } => line.length(),
                        FieldMark::Circle { center: _, radius } => *radius, // approximation
                    };
                    let measured_line_length = transformed_line.length();
                    if measured_line_length <= field_mark_length * line_length_acceptance_factor {
                        let correspondences = field_mark.to_correspondence_points(transformed_line);
                        assert_relative_eq!(
                            correspondences.measured_direction.norm(),
                            1.0,
                            epsilon = 0.0001
                        );
                        assert_relative_eq!(
                            correspondences.reference_direction.norm(),
                            1.0,
                            epsilon = 0.0001
                        );
                        let angle_weight = correspondences
                            .measured_direction
                            .dot(&correspondences.reference_direction)
                            .abs()
                            + measured_line_length / field_mark_length;
                        assert!(field_mark_length != 0.0);
                        let length_weight = measured_line_length / field_mark_length; // TODO: this will penalize center circle lines because field_mark_length is only approximated
                        let weight = angle_weight + length_weight;
                        if weight != 0.0 {
                            Some((correspondences, weight, field_mark, transformed_line))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .min_by_key(
                    |(correspondence_points, weight, _field_mark, _transformed_line)| {
                        assert!(*weight != 0.0);
                        (NotNan::new(
                            distance(
                                &correspondence_points.correspondence_points.0.measured,
                                &correspondence_points.correspondence_points.0.reference,
                            ) + distance(
                                &correspondence_points.correspondence_points.1.measured,
                                &correspondence_points.correspondence_points.1.reference,
                            ),
                        )
                        .unwrap())
                            / *weight
                    },
                )?;
            let inverse_transformation = correction.inverse();
            Some(FieldMarkCorrespondence {
                measured_line_in_field: inverse_transformation * transformed_line,
                field_mark: *field_mark,
                correspondence_points: (
                    CorrespondencePoints {
                        measured: inverse_transformation
                            * correspondences.correspondence_points.0.measured,
                        reference: correspondences.correspondence_points.0.reference,
                    },
                    CorrespondencePoints {
                        measured: inverse_transformation
                            * correspondences.correspondence_points.1.measured,
                        reference: correspondences.correspondence_points.1.reference,
                    },
                ),
            })
        })
        .collect()
}

fn get_correspondence_points(
    field_mark_correspondences: Vec<FieldMarkCorrespondence>,
) -> Vec<CorrespondencePoints> {
    field_mark_correspondences
        .iter()
        .flat_map(|field_mark_correspondence| {
            [
                field_mark_correspondence.correspondence_points.0,
                field_mark_correspondence.correspondence_points.1,
            ]
        })
        .collect()
}

fn get_translation_and_rotation_measurement(
    robot_to_field: Isometry2<f32>,
    field_mark_correspondence: FieldMarkCorrespondence,
) -> Vector2<f32> {
    let (field_mark_line, field_mark_line_direction) = match field_mark_correspondence.field_mark {
        FieldMark::Line { line, direction } => (line, direction),
        _ => panic!("Expected line mark"),
    };
    let measured_line_in_field = match field_mark_line_direction {
        Direction::PositiveX
            if field_mark_correspondence.measured_line_in_field.1.x
                < field_mark_correspondence.measured_line_in_field.0.x =>
        {
            Line(
                field_mark_correspondence.measured_line_in_field.1,
                field_mark_correspondence.measured_line_in_field.0,
            )
        }
        Direction::PositiveY
            if field_mark_correspondence.measured_line_in_field.1.y
                < field_mark_correspondence.measured_line_in_field.0.y =>
        {
            Line(
                field_mark_correspondence.measured_line_in_field.1,
                field_mark_correspondence.measured_line_in_field.0,
            )
        }
        _ => field_mark_correspondence.measured_line_in_field,
    };
    let measured_line_in_field_vector = measured_line_in_field.1 - measured_line_in_field.0;
    let signed_distance_to_line = measured_line_in_field
        .signed_distance_to_point(Point2::from(robot_to_field.translation.vector));
    match field_mark_line_direction {
        Direction::PositiveX => {
            vector![
                field_mark_line.0.y + signed_distance_to_line,
                (-measured_line_in_field_vector.y).atan2(measured_line_in_field_vector.x)
                    + robot_to_field.rotation.angle()
            ]
        }
        Direction::PositiveY => {
            vector![
                field_mark_line.0.x - signed_distance_to_line,
                measured_line_in_field_vector
                    .x
                    .atan2(measured_line_in_field_vector.y)
                    + robot_to_field.rotation.angle()
            ]
        }
    }
}

fn get_2d_translation_measurement(
    robot_to_field: Isometry2<f32>,
    field_mark_correspondence: FieldMarkCorrespondence,
) -> Vector2<f32> {
    let measured_line_vector = field_mark_correspondence.correspondence_points.1.measured
        - field_mark_correspondence.correspondence_points.0.measured;
    let reference_line_vector = field_mark_correspondence.correspondence_points.1.reference
        - field_mark_correspondence.correspondence_points.0.reference;
    let measured_line_point_0_to_robot_vector = Point2::from(robot_to_field.translation.vector)
        - field_mark_correspondence.correspondence_points.0.measured;
    // Signed angle between two vectors: https://wumbo.net/formula/angle-between-two-vectors-2d/
    let measured_rotation = f32::atan2(
        measured_line_point_0_to_robot_vector.y * measured_line_vector.x
            - measured_line_point_0_to_robot_vector.x * measured_line_vector.y,
        measured_line_point_0_to_robot_vector.x * measured_line_vector.x
            + measured_line_point_0_to_robot_vector.y * measured_line_vector.y,
    );

    let reference_line_point_0_to_robot_vector = Rotation2::new(measured_rotation)
        * reference_line_vector.normalize()
        * measured_line_point_0_to_robot_vector.norm();
    let reference_robot_point = field_mark_correspondence.correspondence_points.0.reference
        + reference_line_point_0_to_robot_vector;
    reference_robot_point.coords
}

pub fn generate_initial_pose(
    initial_pose: &InitialPose,
    field_dimensions: &FieldDimensions,
) -> Isometry2<f32> {
    match initial_pose.side {
        Side::Left => Isometry2::new(
            vector!(
                initial_pose.center_line_offset_x,
                field_dimensions.width * 0.5
            ),
            -FRAC_PI_2,
        ),
        Side::Right => Isometry2::new(
            vector!(
                initial_pose.center_line_offset_x,
                -field_dimensions.width * 0.5
            ),
            FRAC_PI_2,
        ),
    }
}

fn generate_penalized_poses(field_dimensions: &FieldDimensions) -> Vec<Isometry2<f32>> {
    vec![
        Isometry2::new(
            vector!(
                -field_dimensions.length * 0.5 + field_dimensions.penalty_marker_distance,
                -field_dimensions.width * 0.5
            ),
            FRAC_PI_2,
        ),
        Isometry2::new(
            vector!(
                -field_dimensions.length * 0.5 + field_dimensions.penalty_marker_distance,
                field_dimensions.width * 0.5
            ),
            -FRAC_PI_2,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_4;

    use nalgebra::point;

    use super::*;

    #[test]
    // test whether signed angle is calculated correctly, not sure why this is in localization
    fn signed_angle() {
        let vector0 = vector![1.0_f32, 0.0_f32];
        let vector1 = vector![0.0_f32, 1.0_f32];
        let vector0_angle = vector0.y.atan2(vector0.x);
        let vector1_angle = vector1.y.atan2(vector1.x);
        assert_relative_eq!(vector1_angle - vector0_angle, FRAC_PI_2);
        assert_relative_eq!(vector0_angle - vector1_angle, -FRAC_PI_2);
    }

    #[test]
    // 
    fn fitting_line_results_in_zero_measurement() {
        let robot_to_field = Isometry2::identity();
        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![0.0, 0.0], point![0.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, Vector2::zeros());

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![0.0, 1.0], point![0.0, 0.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, Vector2::zeros());

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![0.0, 0.0], point![1.0, 0.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, Vector2::zeros());

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 0.0], point![0.0, 0.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, Vector2::zeros());
    }

    #[test]
    fn translated_line_results_in_translation_measurement() {
        let robot_to_field = Isometry2::identity();
        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 0.0], point![1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 1.0], point![1.0, 0.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, 0.0], point![-1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, 1.0], point![-1.0, 0.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![0.0, 1.0], point![1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 1.0], point![0.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![0.0, -1.0], point![1.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![1.0, 0.0]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, -1.0], point![0.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![1.0, 0.0]);
    }

    #[test]
    fn rotated_line_results_in_rotation_measurement() {
        let robot_to_field = Isometry2::identity();
        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, -1.0], point![1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 1.0], point![-1.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, 1.0], point![1.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, -1.0], point![-1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![0.0, -3.0], point![0.0, 3.0]),
                direction: Direction::PositiveY,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, -1.0], point![1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, 1.0], point![-1.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![-1.0, 1.0], point![1.0, -1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, FRAC_PI_4]);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(point![1.0, -1.0], point![-1.0, 1.0]),
            field_mark: FieldMark::Line {
                line: Line(point![-3.0, 0.0], point![3.0, 0.0]),
                direction: Direction::PositiveX,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
                CorrespondencePoints {
                    measured: Point2::origin(),
                    reference: Point2::origin(),
                },
            ),
        };
        let update =
            get_translation_and_rotation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, FRAC_PI_4]);
    }

    #[test]
    fn correct_correspondence_points() {
        let line_length_acceptance_factor = 1.5;

        let measured_lines_in_field = [Line(point![0.0, 0.0], point![1.0, 0.0])];
        let field_marks = [FieldMark::Line {
            line: Line(point![0.0, 0.0], point![1.0, 0.0]),
            direction: Direction::PositiveX,
        }];
        let correspondences = get_field_mark_correspondence(
            &measured_lines_in_field,
            Isometry2::identity(),
            &field_marks,
            line_length_acceptance_factor,
        );
        assert_eq!(correspondences.len(), 1);
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.measured,
            point![0.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.reference,
            point![0.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.measured,
            point![1.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.reference,
            point![1.0, 0.0]
        );

        let measured_lines_in_field = [Line(point![0.0, 0.0], point![1.0, 0.0])];
        let field_marks = [FieldMark::Line {
            line: Line(point![0.0, 1.0], point![1.0, 1.0]),
            direction: Direction::PositiveX,
        }];
        let correspondences = get_field_mark_correspondence(
            &measured_lines_in_field,
            Isometry2::identity(),
            &field_marks,
            line_length_acceptance_factor,
        );
        assert_eq!(correspondences.len(), 1);
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.measured,
            point![0.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.reference,
            point![0.0, 1.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.measured,
            point![1.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.reference,
            point![1.0, 1.0]
        );

        let measured_lines_in_field = [Line(point![0.0, 0.0], point![1.0, 0.0])];
        let field_marks = [FieldMark::Line {
            line: Line(point![0.0, 0.0], point![1.0, 0.0]),
            direction: Direction::PositiveX,
        }];
        let correspondences = get_field_mark_correspondence(
            &measured_lines_in_field,
            Isometry2::new(vector![0.0, 1.0], 0.0),
            &field_marks,
            line_length_acceptance_factor,
        );
        assert_eq!(correspondences.len(), 1);
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.measured,
            point![0.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.0.reference,
            point![0.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.measured,
            point![1.0, 0.0]
        );
        assert_relative_eq!(
            correspondences[0].correspondence_points.1.reference,
            point![1.0, 0.0]
        );
    }

    #[test]
    fn circle_mark_correspondence_translates() {
        let robot_to_field = Isometry2::identity();
        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![0.0, 0.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![1.0, 0.0],
                    reference: point![1.0, 0.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, 0.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![0.0, 1.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![1.0, 1.0],
                    reference: point![1.0, 0.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -1.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![0.0, -1.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![1.0, -1.0],
                    reference: point![1.0, 0.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, 1.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![1.0, 0.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![1.0, 1.0],
                    reference: point![0.0, 1.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, 0.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![-1.0, 0.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![-1.0, 1.0],
                    reference: point![0.0, 1.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![1.0, 0.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![1.0, 1.0],
                    reference: point![0.0, 0.0],
                },
                CorrespondencePoints {
                    measured: point![1.0, 2.0],
                    reference: point![0.0, 1.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![-1.0, -1.0], epsilon = 0.0001);

        let field_mark_correspondence = FieldMarkCorrespondence {
            measured_line_in_field: Line(Point2::origin(), Point2::origin()),
            field_mark: FieldMark::Circle {
                center: Point2::origin(),
                radius: 0.0,
            },
            correspondence_points: (
                CorrespondencePoints {
                    measured: point![1.0, 1.0],
                    reference: point![-1.0, -1.0],
                },
                CorrespondencePoints {
                    measured: point![2.0, 1.0],
                    reference: point![-1.0, 0.0],
                },
            ),
        };
        let update = get_2d_translation_measurement(robot_to_field, field_mark_correspondence);
        assert_relative_eq!(update, vector![0.0, -2.0], epsilon = 0.0001);
    }
}
