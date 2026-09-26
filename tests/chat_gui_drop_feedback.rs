use std::fs;

fn gui_source() -> String {
    fs::read_to_string("src/bin/syntrail_gui.rs").expect("read syntrail_gui.rs")
}

fn excerpt_after(source: &str, marker: &str, chars: usize) -> String {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing marker: {marker}"));
    source[start..].chars().take(chars).collect()
}

#[test]
fn multiple_chat_drop_files_report_a_visible_status() {
    let source = gui_source();
    let tail = excerpt_after(&source, "DropResult::MultipleFiles", 500);

    assert!(
        tail.contains("self.status"),
        "multiple-file drops must update visible status"
    );
    assert!(
        tail.contains("one model file") || tail.contains("1 model file"),
        "status must explain the one-model-file Chat limit"
    );
}

#[test]
fn unsupported_chat_drop_reports_the_rejected_path_and_reason() {
    let source = gui_source();
    let tail = excerpt_after(&source, "DropResult::Unsupported", 700);

    assert!(
        tail.contains("self.status"),
        "unsupported drops must update visible status"
    );
    assert!(
        tail.contains("display()"),
        "status must identify the rejected path"
    );
    assert!(
        tail.contains("unsupported")
            || tail.contains("not supported")
            || tail.contains("model file"),
        "status must explain why the file was rejected"
    );
}

#[test]
fn chat_drop_feedback_does_not_change_router_acceptance_rules() {
    let source = fs::read_to_string("src/desktop/drop.rs").expect("read drop.rs");
    assert!(
        source.contains("Self { model: true, dataset: false }"),
        "Chat must remain model-only; this maintenance is feedback-only"
    );
}
