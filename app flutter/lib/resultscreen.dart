import 'dart:io';
import 'package:flutter/material.dart';
import 'shared.dart';
import 'package:google_fonts/google_fonts.dart';

class ResultScreen extends StatefulWidget {
  final PredictionResult result;
  final File originalImage;
  final String? patientName;
  final String? patientAge;
  final String? gender;

  const ResultScreen({
    super.key, 
    required this.result, 
    required this.originalImage,
    this.patientName,
    this.patientAge,
    this.gender,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  int _viewIndex = 2; // 0: Gốc, 1: Heatmap, 2: Overlay
  bool _showGrid = true;

  @override
  Widget build(BuildContext context) {
    final color = widget.result.isPneumonia ? kRed : kGreen;

    return Scaffold(
      backgroundColor: kBg,
      appBar: AppBar(
        backgroundColor: kBg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        title: Text(
          "Chi tiết chẩn đoán",
          style: GoogleFonts.poppins(fontWeight: FontWeight.w700, fontSize: 18),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (widget.patientName != null || widget.patientAge != null || widget.gender != null) ...[
              _buildPatientInfo(),
              const SizedBox(height: 16),
            ],
            _buildResultHeader(color),
            const SizedBox(height: 24),
            
            _buildVisualizerSection(),
            const SizedBox(height: 24),
            
            _buildExecutionTimes(),

            const SizedBox(height: 24),
            const Divider(color: kBorder),
            const SizedBox(height: 16),
            const Center(
              child: Text(
                "LƯU Ý ĐẠO ĐỨC & PHÁP LÝ: Kết quả từ AI chỉ mang tính chất tham khảo và hỗ trợ sàng lọc sơ bộ. Quyết định điều trị và chẩn đoán cuối cùng phải dựa trên kết luận chuyên môn của bác sĩ chuyên khoa.",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 11,
                  color: kTextSecond,
                  fontStyle: FontStyle.italic,
                  height: 1.5,
                ),
              ),
            ),

            const SizedBox(height: 32),
            _buildBackButton(context),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _buildPatientInfo() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.person_pin_rounded, size: 20, color: kCyan),
              const SizedBox(width: 8),
              const Text("Thông tin bệnh nhân", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: kTextPrim)),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              if (widget.patientName != null)
                Expanded(
                  child: _infoItem("Họ tên", widget.patientName!),
                ),
              if (widget.patientAge != null)
                SizedBox(
                  width: 60,
                  child: _infoItem("Tuổi", widget.patientAge!),
                ),
              if (widget.gender != null)
                SizedBox(
                  width: 80,
                  child: _infoItem("Giới tính", widget.gender!),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _infoItem(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: kTextSecond)),
        const SizedBox(height: 2),
        Text(value, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: kTextPrim)),
      ],
    );
  }

  Widget _buildResultHeader(Color color) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.2), width: 1.5),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.result.isPneumonia ? "PHÁT HIỆN VIÊM PHỔI" : "PHỔI BÌNH THƯỜNG", 
                  style: TextStyle(color: color, fontWeight: FontWeight.w800, fontSize: 17)
                ),
                const SizedBox(height: 4),
                Text(
                  widget.result.isPneumonia 
                      ? "Cần tham khảo ý kiến bác sĩ chuyên khoa."
                      : "Không tìm thấy dấu hiệu thâm nhiễm.", 
                  style: const TextStyle(color: kTextSecond, fontSize: 12)
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          CircularPercentIndicator(
            percent: widget.result.confidence,
            color: color,
            text: "${(widget.result.confidence * 100).toStringAsFixed(0)}%",
          ),
        ],
      ),
    );
  }

  Widget _buildVisualizerSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.auto_awesome, size: 18, color: kCyan),
            const SizedBox(width: 8),
            const Text("Giải thích từ mô hình AI (Grad-CAM):", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 16),
        
        // Horizontal Scroll for 3 panels
        SizedBox(
          height: 240,
          child: ListView(
            scrollDirection: Axis.horizontal,
            children: [
              _buildImagePanel("ẢNH GỐC", widget.result.imageUrl, isFile: widget.result.imageUrl == null),
              _buildImagePanel("BẢN ĐỒ NHIỆT", widget.result.heatmapUrl),
              _buildImagePanel("OVERLAY (FOCUS)", widget.result.gradcamUrl),
            ],
          ),
        ),
        
        const SizedBox(height: 12),
        const Center(
          child: Text(
            "Vuốt sang phải để xem chi tiết các bản đồ nhiệt", 
            style: TextStyle(fontSize: 11, color: kTextSecond, fontStyle: FontStyle.italic)
          ),
        ),
      ],
    );
  }

  Widget _buildImagePanel(String title, String? url, {bool isFile = false}) {
    return Container(
      width: 200,
      margin: const EdgeInsets.only(right: 16),
      decoration: BoxDecoration(
        color: kSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: kBorder),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(title, style: GoogleFonts.inter(fontSize: 10, fontWeight: FontWeight.w800, color: kTextSecond)),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(15)),
              child: isFile 
                ? Image.file(widget.originalImage, width: double.infinity, fit: BoxFit.cover)
                : (url != null 
                    ? Image.network(url, width: double.infinity, fit: BoxFit.cover, 
                        errorBuilder: (_, __, ___) => const Center(child: Icon(Icons.broken_image)))
                    : const Center(child: CircularProgressIndicator(strokeWidth: 2))),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildViewTab(int index, String label) {
    bool isSelected = _viewIndex == index;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _viewIndex = index),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isSelected ? Colors.white : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
            boxShadow: isSelected ? [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 4)] : null,
          ),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
              color: isSelected ? kCyan : kTextSecond,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageByView() {
    String? url;
    if (_viewIndex == 0) {
      if (widget.result.imageUrl != null) return Image.network(widget.result.imageUrl!, width: double.infinity, height: 320, fit: BoxFit.cover);
      return Image.file(widget.originalImage, width: double.infinity, height: 320, fit: BoxFit.cover);
    } else if (_viewIndex == 1) {
      url = widget.result.heatmapUrl;
    } else {
      url = widget.result.gradcamUrl;
    }

    if (url == null) return Container(height: 320, color: kSurface, child: const Center(child: Text("Không có dữ liệu")));

    return Image.network(
      url, 
      width: double.infinity, 
      height: 320, 
      fit: BoxFit.cover,
      errorBuilder: (_, __, ___) => Container(height: 320, color: kSurface, child: const Icon(Icons.broken_image)),
    );
  }

  Widget _gridZone(String label) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(border: Border.all(color: Colors.white.withValues(alpha: 0.1), width: 0.3)),
        child: Center(
          child: Text(label, style: TextStyle(color: Colors.white.withValues(alpha: 0.4), fontSize: 12, fontWeight: FontWeight.bold)),
        ),
      ),
    );
  }

  Widget _buildProbSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Xác suất chi tiết", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        _simpleProbBar("Bình thường (Normal)", widget.result.normalProb, kGreen),
        const SizedBox(height: 12),
        _simpleProbBar("Viêm phổi (Pneumonia)", widget.result.pneumoniaProb, kRed),
      ],
    );
  }

  Widget _buildSeveritySection() {
    final s = widget.result.severity!;
    Color severityColor = s.gradeEn == "Severe" ? kRed : (s.gradeEn == "Moderate" ? Colors.orange : Colors.amber);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: kBorder),
        boxShadow: [
          BoxShadow(color: Colors.black.withValues(alpha: 0.03), blurRadius: 15, offset: const Offset(0, 5)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: severityColor.withValues(alpha: 0.1), shape: BoxShape.circle),
                child: Icon(Icons.analytics_rounded, color: severityColor, size: 20),
              ),
              const SizedBox(width: 12),
              const Text("Phân tích mức độ", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const Spacer(),
              _Badge(text: s.gradeVi.toUpperCase(), color: severityColor),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              _severityMetric("Điểm Brixia", "${s.brixiaScore}/18", severityColor),
              const Spacer(),
              _severityMetric("Vùng ảnh hưởng", "${s.affectedAreaPct.toStringAsFixed(1)}%", kCyan),
            ],
          ),
          const SizedBox(height: 24),
          const Text("Phân bổ thâm nhiễm (Brixia 6 zones):", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: kTextSecond)),
          const SizedBox(height: 12),
          _buildBrixiaGrid(s.zoneScores, severityColor),
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: kSurface,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(
              s.gradeDescription,
              style: const TextStyle(fontSize: 13, color: kTextPrim, height: 1.5),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBrixiaGrid(List<int> scores, Color color) {
    if (scores.length < 6) return const SizedBox();
    return Row(
      children: [
        Expanded(
          child: Column(
            children: [
              const Text("Phổi Phải (Right)", style: TextStyle(fontSize: 10, color: kTextSecond)),
              const SizedBox(height: 6),
              _zoneBox(scores[0], color),
              _zoneBox(scores[1], color),
              _zoneBox(scores[2], color),
            ],
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            children: [
              const Text("Phổi Trái (Left)", style: TextStyle(fontSize: 10, color: kTextSecond)),
              const SizedBox(height: 6),
              _zoneBox(scores[3], color),
              _zoneBox(scores[4], color),
              _zoneBox(scores[5], color),
            ],
          ),
        ),
      ],
    );
  }

  Widget _zoneBox(int score, Color baseColor) {
    Color boxColor = score == 0 ? kSurface : (score == 1 ? baseColor.withValues(alpha: 0.3) : (score == 2 ? baseColor.withValues(alpha: 0.6) : baseColor));
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 2),
      height: 32,
      width: double.infinity,
      decoration: BoxDecoration(color: boxColor, borderRadius: BorderRadius.circular(8)),
      child: Center(
        child: Text(score.toString(), style: TextStyle(color: score >= 2 ? Colors.white : kTextPrim, fontWeight: FontWeight.bold, fontSize: 13)),
      ),
    );
  }

  Widget _severityMetric(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: kTextSecond)),
        const SizedBox(height: 2),
        Text(value, style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900, color: color)),
      ],
    );
  }

  Widget _simpleProbBar(String label, double val, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontSize: 12, color: kTextPrim, fontWeight: FontWeight.w500)),
            Text("${(val * 100).toStringAsFixed(1)}%", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: color)),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: val,
            backgroundColor: kBorder,
            color: color,
            minHeight: 8,
          ),
        ),
      ],
    );
  }

  Widget _buildExecutionTimes() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.timer_outlined, size: 20, color: kCyan),
              const SizedBox(width: 8),
              Text(
                "Hiệu năng xử lý AI",
                style: GoogleFonts.poppins(
                  fontSize: 14, 
                  fontWeight: FontWeight.bold, 
                  color: kTextPrim
                )
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("Tiền xử lý ảnh", style: TextStyle(fontSize: 10, color: kTextSecond)),
                    const SizedBox(height: 2),
                    Text(
                      widget.result.preprocessTime != null 
                          ? "${(widget.result.preprocessTime! * 1000).toStringAsFixed(1)} ms"
                          : "N/A", 
                      style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: kTextPrim)
                    ),
                  ],
                ),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("Suy luận AI", style: TextStyle(fontSize: 10, color: kTextSecond)),
                    const SizedBox(height: 2),
                    Text(
                      widget.result.inferenceTime != null 
                          ? "${(widget.result.inferenceTime! * 1000).toStringAsFixed(1)} ms"
                          : "N/A", 
                      style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: kTextPrim)
                    ),
                  ],
                ),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("Tạo Grad-CAM", style: TextStyle(fontSize: 10, color: kTextSecond)),
                    const SizedBox(height: 2),
                    Text(
                      widget.result.gradcamTime != null 
                          ? "${(widget.result.gradcamTime! * 1000).toStringAsFixed(1)} ms"
                          : "N/A", 
                      style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: kTextPrim)
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBackButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: () => Navigator.pop(context),
        style: ElevatedButton.styleFrom(
          backgroundColor: kTextPrim,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 18),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          elevation: 0,
        ),
        child: const Text("Xác nhận & Quay lại", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final String text;
  final Color color;
  const _Badge({required this.text, required this.color});
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(8)),
      child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
    );
  }
}

class CircularPercentIndicator extends StatelessWidget {
  final double percent;
  final Color color;
  final String text;
  const CircularPercentIndicator({super.key, required this.percent, required this.color, required this.text});

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        SizedBox(
          width: 50,
          height: 50,
          child: CircularProgressIndicator(
            value: percent,
            strokeWidth: 5,
            color: color,
            backgroundColor: color.withValues(alpha: 0.1),
          ),
        ),
        Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: color)),
      ],
    );
  }
}
