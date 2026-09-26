use std::fs;

fn gui_source() -> String {
    fs::read_to_string("src/bin/syntrail_gui.rs").expect("read syntrail_gui.rs")
}

#[test]
fn multiple_chat_drop_files_report_a_visible_status() {
    let source = gui_source();
    let multiple = source
        .find("DropResult::MultipleFiles")
        .expect("Chat GUI must handle MultipleFiles explicitly");
    let tail = &source[multiple..source.len().min(multiple + 500)];

    assert!(tail.contains("self.status"), "multiple-file drops must update visible status");
    assert!(tail.contains("one model file") || tail.contains("1 model file"),
        "status must explain the one-model-file Chat limit");
}

#[test]
fn unsupported_chat_drop_reports_the_rejected_path_and_reason() {
    let source = gui_source();
    let unsupported = source
        .find("DropResult::Unsupported")
        .expect("Chat GUI must handle Unsupported explicitly");
    let tail = &source[unsupported..source.len().min(unsupported + 700)];

    assert!(tail.contains("self.status"), "unsupported drops must update visible status");
    assert!(tail.contains("display()"), "status must identify the rejected path");
    assert!(tail.contains("unsupported") || tail.contains("not supported") || tail.contains("model file"),
        "status must explain why the file was rejected");
}

#[test]
fn chat_drop_feedback_does_not_change_router_acceptance_rules() {
    let source = fs::read_to_string("src/desktop/drop.rs").expect("read drop.rs");
    assert!(source.contains("Self { model: true, dataset: false }"),
        "Chat must remain model-only; this maintenance is feedback-only");
}
