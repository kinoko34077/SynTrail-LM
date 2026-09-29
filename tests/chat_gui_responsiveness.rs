use std::fs;
use syntrail_lm::desktop::chat_state::{
    ChatLifecycle, CloseRequestDecision, WorkerExit, close_request_decision,
    lifecycle_after_worker_exit, restart_required_message,
};

fn gui_source() -> String {
    fs::read_to_string("src/bin/syntrail_gui.rs").expect("read syntrail_gui.rs")
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index + start.len()..]
        .find(end)
        .map(|offset| start_index + start.len() + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

#[test]
fn lifecycle_behavior_blocks_duplicate_or_crashed_model_actions() {
    assert!(ChatLifecycle::Idle.can_send());
    assert!(ChatLifecycle::Idle.can_mutate_model());

    for blocked in [ChatLifecycle::Generating, ChatLifecycle::RestartRequired] {
        assert!(!blocked.can_send());
        assert!(!blocked.can_mutate_model());
    }
}

#[test]
fn worker_panic_and_disconnect_require_restart_but_normal_failures_do_not() {
    assert_eq!(
        lifecycle_after_worker_exit(WorkerExit::Completed),
        ChatLifecycle::Idle
    );
    assert_eq!(
        lifecycle_after_worker_exit(WorkerExit::Failed),
        ChatLifecycle::Idle
    );
    for crashed in [WorkerExit::Panicked, WorkerExit::Disconnected] {
        assert_eq!(
            lifecycle_after_worker_exit(crashed),
            ChatLifecycle::RestartRequired
        );
        assert!(
            restart_required_message(crashed)
                .unwrap()
                .contains("Restart the app")
        );
    }
}

#[test]
fn close_decision_is_behavioral_for_generating_and_restart_required_states() {
    assert_eq!(
        close_request_decision(ChatLifecycle::Generating, false, false),
        CloseRequestDecision::Defer
    );
    assert_eq!(
        close_request_decision(ChatLifecycle::RestartRequired, false, false),
        CloseRequestDecision::Allow
    );
    assert_eq!(
        close_request_decision(ChatLifecycle::RestartRequired, true, false),
        CloseRequestDecision::Cancel
    );
    assert_eq!(
        close_request_decision(ChatLifecycle::RestartRequired, true, true),
        CloseRequestDecision::Allow
    );
}

#[test]
fn send_dispatch_does_not_generate_on_the_egui_update_thread() {
    let source = gui_source();
    assert!(source.contains("thread::spawn") || source.contains("std::thread::spawn"));
    assert!(source.contains("try_recv"));
    assert!(!source.contains("Action::Send            => self.send_message()"));
}

#[test]
fn invalid_or_multiple_chat_drop_reports_visible_status() {
    let source = gui_source();
    let drop_block = section(&source, "match route_drop", "let model_path_str");
    assert!(drop_block.contains("DropResult::MultipleFiles =>"));
    assert!(drop_block.contains("Drop rejected: multiple files are not supported in Chat"));
    assert!(drop_block.contains("DropResult::Unsupported(path) =>"));
    assert!(drop_block.contains("Unsupported dropped file"));
}

#[test]
fn completed_turn_commits_the_already_computed_analytics_snapshot_immediately() {
    let source = gui_source();
    let completion = section(
        &source,
        "GenerationOutcome::Completed(turn_id, output) => {",
        "GenerationOutcome::Failed(error) => {",
    );
    assert!(completion.contains("self.analytics = a;"));
    assert!(!source.contains("turns_since_refresh"));
    assert!(!source.contains("refresh_interval"));
}

#[test]
fn restart_required_state_blocks_deferred_model_actions() {
    let source = gui_source();
    assert!(
        source.contains("self.lifecycle().restart_required() && !matches!(action, Action::None)"),
        "RestartRequired must stop toolbar/native-menu model actions before dispatch"
    );
}
