/// CJK font setup shared across all GUI applications (§122).
#[cfg(feature = "gui")]
pub fn setup_fonts(ctx: &eframe::egui::Context) {
    let candidates: &[&str] = &[
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\msmincho.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    ];
    let Some(data) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut fonts = eframe::egui::FontDefinitions::default();
    fonts.font_data.insert("cjk".to_owned(), eframe::egui::FontData::from_owned(data));
    for family in [
        eframe::egui::FontFamily::Proportional,
        eframe::egui::FontFamily::Monospace,
    ] {
        fonts.families.entry(family).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
}
