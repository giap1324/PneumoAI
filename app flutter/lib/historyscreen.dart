import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:google_fonts/google_fonts.dart';

// ─── Constants ───────────────────────────────────────────────────────────────
const Color _white       = Colors.white;
const Color _bg          = Color(0xFFF8F9FB);
const Color _card        = Colors.white;
const Color _border      = Color(0xFFEEEEF0);
const Color _textPri     = Color(0xFF111827);
const Color _textSec     = Color(0xFF9CA3AF);
const Color _blue        = Color(0xFF2563EB);
const Color _green       = Color(0xFF16A34A);
const Color _red         = Color(0xFFDC2626);

// ─── Model ───────────────────────────────────────────────────────────────────
class HistoryEntry {
  final int id;
  final String imageUrl;
  final String? heatmapUrl;
  final String? gradcamUrl;
  final String label;
  final double confidence;
  final double normalProb;
  final double pneumoniaProb;
  final String? patientName;
  final String? patientAge;
  final String? gender;
  final DateTime timestamp;

  HistoryEntry({
    required this.id,
    required this.imageUrl,
    this.heatmapUrl,
    this.gradcamUrl,
    required this.label,
    required this.confidence,
    required this.normalProb,
    required this.pneumoniaProb,
    this.patientName,
    this.patientAge,
    this.gender,
    required this.timestamp,
  });

  factory HistoryEntry.fromJson(Map<String, dynamic> json, String baseUrl) {
    return HistoryEntry(
      id: json['id'],
      imageUrl: baseUrl + (json['image_url'] ?? ''),
      heatmapUrl: json['heatmap_url'] != null ? baseUrl + json['heatmap_url'] : null,
      gradcamUrl: json['gradcam_url'] != null ? baseUrl + json['gradcam_url'] : null,
      label: json['label'] ?? '',
      confidence: (json['confidence'] as num).toDouble(),
      normalProb: (json['normal_prob'] as num).toDouble(),
      pneumoniaProb: (json['pneumonia_prob'] as num).toDouble(),
      patientName: json['patient_name'],
      patientAge: json['patient_age'],
      gender: json['gender'],
      timestamp: DateTime.parse(json['timestamp']),
    );
  }
}

// ─── Store ───────────────────────────────────────────────────────────────────
class HistoryStore extends ChangeNotifier {
  static final HistoryStore _instance = HistoryStore._();
  HistoryStore._();
  factory HistoryStore() => _instance;

  final String baseUrl = "http://10.0.2.2:8000";

  List<HistoryEntry> _items = [];
  bool _loading = false;
  String? _error;

  List<HistoryEntry> get items => _items;
  bool get isLoading => _loading;
  String? get error => _error;

  Future<void> fetchHistory() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      final response = await http.get(Uri.parse("$baseUrl/history"));
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        _items = data.map((j) => HistoryEntry.fromJson(j, baseUrl)).toList();
      } else {
        _error = "Lỗi server: ${response.statusCode}";
      }
    } catch (e) {
      _error = "Lỗi kết nối: $e";
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> delete(int id) async {
    try {
      final response = await http.delete(Uri.parse("$baseUrl/history/$id"));
      if (response.statusCode == 200) {
        _items.removeWhere((item) => item.id == id);
        notifyListeners();
      }
    } catch (e) {
      debugPrint("Delete error: $e");
    }
  }
}

// ─── Screen ──────────────────────────────────────────────────────────────────
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      HistoryStore().fetchHistory();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: _buildAppBar(),
      body: ListenableBuilder(
        listenable: HistoryStore(),
        builder: (context, _) {
          final store = HistoryStore();

          if (store.isLoading && store.items.isEmpty) {
            return const Center(
              child: CircularProgressIndicator(color: _blue, strokeWidth: 2),
            );
          }

          if (store.error != null && store.items.isEmpty) {
            return _buildErrorState(store);
          }

          if (store.items.isEmpty) {
            return _buildEmptyState();
          }

          final total     = store.items.length;
          final normal    = store.items.where((e) => e.label == "NORMAL").length;
          final pneumonia = store.items.where((e) => e.label == "PNEUMONIA").length;

          return RefreshIndicator(
            color: _blue,
            onRefresh: () => store.fetchHistory(),
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 110),
              children: [
                _buildSummaryRow(total, normal, pneumonia),
                const SizedBox(height: 24),
                _buildSectionHeader("Danh sách chẩn đoán", total),
                const SizedBox(height: 12),
                ...store.items.map((e) => _buildCard(e)),
              ],
            ),
          );
        },
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      backgroundColor: Colors.white.withValues(alpha: 0.5),
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      centerTitle: false,
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(1),
        child: Container(height: 1, color: _border),
      ),
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Logo
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(11),
              gradient: const LinearGradient(
                colors: [Color(0xFF2563EB), Color(0xFF7C3AED)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF2563EB).withValues(alpha: 0.28),
                  blurRadius: 12,
                  spreadRadius: 0,
                  offset: const Offset(0, 4),
                ),
                BoxShadow(
                  color: const Color(0xFF7C3AED).withValues(alpha: 0.15),
                  blurRadius: 20,
                  spreadRadius: -2,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Vòng tròn mờ trang trí
                Positioned(
                  top: -4,
                  right: -4,
                  child: Container(
                    width: 20,
                    height: 20,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withValues(alpha: 0.12),
                    ),
                  ),
                ),
                // Icon chính
                const Icon(
                  Icons.monitor_heart_outlined,
                  color: Colors.white,
                  size: 20,
                ),
              ],
            ),
          ),

          const SizedBox(width: 12),

          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ShaderMask(
                blendMode: BlendMode.srcIn,
                shaderCallback: (bounds) => const LinearGradient(
                  colors: [Color(0xFF2563EB), Color(0xFF7C3AED)],
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                ).createShader(bounds),
                child: Text(
                  "Lịch sử chẩn đoán",
                  style: GoogleFonts.inter(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.4,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 1),
              Row(
                children: [
                  Container(
                    width: 5,
                    height: 5,
                    decoration: const BoxDecoration(
                      shape: BoxShape.circle,
                      color: Color(0xFF22C55E),
                    ),
                  ),
                  const SizedBox(width: 5),
                  Text(
                    "Pneumonia Detection AI",
                    style: GoogleFonts.inter(
                      fontSize: 11,
                      fontWeight: FontWeight.w400,
                      color: _textSec,
                      letterSpacing: 0.1,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
      actions: [
        Container(
          margin: const EdgeInsets.only(right: 16),
          child: IconButton(
            icon: const Icon(Icons.refresh_rounded, size: 18),
            onPressed: () => HistoryStore().fetchHistory(),
            style: IconButton.styleFrom(
              foregroundColor: const Color(0xFF2563EB),
              backgroundColor: const Color(0xFF2563EB).withOpacity(0.08),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
                side: BorderSide(
                  color: const Color(0xFF2563EB).withOpacity(0.15),
                ),
              ),
              padding: const EdgeInsets.all(9),
            ),
          ),
        ),
      ],
    );

  }

  // ── Summary Row ───────────────────────────────────────────────────────────
  Widget _buildSummaryRow(int total, int normal, int pneumonia) {
    return Row(
      children: [
        Expanded(child: _StatCard(label: "Tổng số", value: total, icon: Icons.bar_chart_rounded, color: _blue)),
        const SizedBox(width: 10),
        Expanded(child: _StatCard(label: "Bình thường", value: normal, icon: Icons.check_circle_outline_rounded, color: _green)),
        const SizedBox(width: 10),
        Expanded(child: _StatCard(label: "Viêm phổi", value: pneumonia, icon: Icons.warning_amber_rounded, color: _red)),
      ],
    );
  }

  // ── Section header ────────────────────────────────────────────────────────
  Widget _buildSectionHeader(String title, int count) {
    return Row(
      children: [
        Text(
          title,
          style: GoogleFonts.inter(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: _textPri,
          ),
        ),
        const SizedBox(width: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          decoration: BoxDecoration(
            color: _blue.withOpacity(0.08),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            count.toString(),
            style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: _blue),
          ),
        ),
      ],
    );
  }

  // ── Entry Card ────────────────────────────────────────────────────────────
  Widget _buildCard(HistoryEntry entry) {
    final isPneumonia = entry.label == "PNEUMONIA";
    final color       = isPneumonia ? _red : _green;
    final labelText   = isPneumonia ? "Viêm phổi" : "Bình thường";
    final ts          = entry.timestamp;
    final timeStr     = "${ts.day.toString().padLeft(2, '0')}/${ts.month.toString().padLeft(2, '0')}/${ts.year}  ${ts.hour.toString().padLeft(2, '0')}:${ts.minute.toString().padLeft(2, '0')}";

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: _card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            // Thumbnail
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.network(
                entry.imageUrl,
                width: 64,
                height: 64,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(
                  width: 64,
                  height: 64,
                  color: _bg,
                  child: const Icon(Icons.image_not_supported_outlined, color: _textSec, size: 22),
                ),
              ),
            ),
            const SizedBox(width: 14),

            // Info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      // Badge
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: color.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              isPneumonia ? Icons.warning_amber_rounded : Icons.check_circle_outline_rounded,
                              size: 12,
                              color: color,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              labelText,
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                color: color,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const Spacer(),
                      Text(
                        "${(entry.confidence * 100).toStringAsFixed(1)}%",
                        style: GoogleFonts.inter(
                          fontSize: 15,
                          fontWeight: FontWeight.w800,
                          color: color,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 10),

                  // Progress bar
                  ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: entry.confidence,
                      minHeight: 4,
                      backgroundColor: _border,
                      color: color,
                    ),
                  ),

                  const SizedBox(height: 10),

                  Row(
                    children: [
                      const Icon(Icons.access_time_rounded, size: 12, color: _textSec),
                      const SizedBox(width: 4),
                      Text(
                        timeStr,
                        style: GoogleFonts.inter(fontSize: 11, color: _textSec),
                      ),
                      const Spacer(),

                      // Detail button
                      GestureDetector(
                        onTap: () => _showDetail(entry),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: _blue.withOpacity(0.07),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                "Chi tiết",
                                style: GoogleFonts.inter(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: _blue,
                                ),
                              ),
                              const SizedBox(width: 3),
                              const Icon(Icons.arrow_forward_rounded, size: 11, color: _blue),
                            ],
                          ),
                        ),
                      ),

                      const SizedBox(width: 8),

                      // Delete button
                      GestureDetector(
                        onTap: () => _confirmDelete(entry),
                        child: Container(
                          padding: const EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: _red.withOpacity(0.06),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: const Icon(Icons.delete_outline_rounded, size: 15, color: _red),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Empty / Error ─────────────────────────────────────────────────────────
  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: _bg,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _border),
            ),
            child: const Icon(Icons.inbox_outlined, color: _textSec, size: 32),
          ),
          const SizedBox(height: 16),
          Text("Chưa có dữ liệu", style: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600, color: _textPri)),
          const SizedBox(height: 6),
          Text("Các lần chẩn đoán sẽ xuất hiện ở đây", style: GoogleFonts.inter(fontSize: 13, color: _textSec)),
        ],
      ),
    );
  }

  Widget _buildErrorState(HistoryStore store) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: _red.withOpacity(0.06),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Icon(Icons.wifi_off_rounded, color: _red, size: 32),
          ),
          const SizedBox(height: 16),
          Text("Không thể kết nối", style: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600, color: _textPri)),
          const SizedBox(height: 6),
          Text(store.error ?? "", style: GoogleFonts.inter(fontSize: 12, color: _textSec)),
          const SizedBox(height: 20),
          FilledButton.icon(
            onPressed: () => store.fetchHistory(),
            icon: const Icon(Icons.refresh_rounded, size: 16),
            label: const Text("Thử lại"),
            style: FilledButton.styleFrom(
              backgroundColor: _blue,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
          ),
        ],
      ),
    );
  }

  // ── Delete Dialog ─────────────────────────────────────────────────────────
  void _confirmDelete(HistoryEntry entry) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: _white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text("Xóa bản ghi?", style: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w700, color: _textPri)),
        content: Text("Hành động này không thể hoàn tác.", style: GoogleFonts.inter(fontSize: 13, color: _textSec)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text("Hủy", style: GoogleFonts.inter(color: _textSec, fontWeight: FontWeight.w500)),
          ),
          FilledButton(
            onPressed: () {
              HistoryStore().delete(entry.id);
              Navigator.pop(ctx);
            },
            style: FilledButton.styleFrom(
              backgroundColor: _red,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
            child: Text("Xóa", style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  // ── Detail Bottom Sheet ───────────────────────────────────────────────────
  void _showDetail(HistoryEntry entry) {
    final isPneumonia = entry.label == "PNEUMONIA";
    final color       = isPneumonia ? _red : _green;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => DraggableScrollableSheet(
        initialChildSize: 0.75,
        maxChildSize: 0.95,
        minChildSize: 0.4,
        expand: false,
        builder: (ctx, scrollCtrl) => Container(
          decoration: const BoxDecoration(
            color: _white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              // Handle
              Padding(
                padding: const EdgeInsets.only(top: 12, bottom: 4),
                child: Container(
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                    color: _border,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),

              // Header
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 12, 24, 0),
                child: Row(
                  children: [
                    Text(
                      "Chi tiết chẩn đoán",
                      style: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w700, color: _textPri),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        isPneumonia ? "Viêm phổi" : "Bình thường",
                        style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: color),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 4),
              Container(margin: const EdgeInsets.only(top: 12), height: 1, color: _border),

              // Scrollable content
              Expanded(
                child: SingleChildScrollView(
                  controller: scrollCtrl,
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Patient Info
                      if (entry.patientName != null || entry.patientAge != null || entry.gender != null) ...[
                        _DetailSectionLabel("Thông tin bệnh nhân"),
                        const SizedBox(height: 10),
                        Container(
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: _bg,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: _border),
                          ),
                          child: Column(
                            children: [
                              if (entry.patientName != null)
                                _buildDetailRow(Icons.person_outline_rounded, "Họ tên", entry.patientName!),
                              if (entry.patientAge != null) ...[
                                if (entry.patientName != null) const Divider(height: 20),
                                _buildDetailRow(Icons.calendar_today_rounded, "Tuổi", entry.patientAge!),
                              ],
                              if (entry.gender != null) ...[
                                if (entry.patientName != null || entry.patientAge != null) const Divider(height: 20),
                                _buildDetailRow(Icons.wc_rounded, "Giới tính", entry.gender!),
                              ],
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],

                      // X-Ray image
                      _DetailSectionLabel("Ảnh X-quang"),
                      const SizedBox(height: 10),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(14),
                        child: Image.network(
                          entry.imageUrl,
                          width: double.infinity,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(
                            height: 180,
                            color: _bg,
                            child: const Center(child: Icon(Icons.image_not_supported_outlined, color: _textSec, size: 32)),
                          ),
                        ),
                      ),

                      // Grad-CAM
                      if (entry.gradcamUrl != null) ...[
                        const SizedBox(height: 24),
                        _DetailSectionLabel("Ảnh Overlay (Giải thích AI)"),
                        const SizedBox(height: 10),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(14),
                          child: Image.network(
                            entry.gradcamUrl!,
                            width: double.infinity,
                            fit: BoxFit.cover,
                          ),
                        ),
                      ],

                      // Heatmap
                      if (entry.heatmapUrl != null) ...[
                        const SizedBox(height: 24),
                        _DetailSectionLabel("Bản đồ nhiệt (Heatmap)"),
                        const SizedBox(height: 10),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(14),
                          child: Image.network(
                            entry.heatmapUrl!,
                            width: double.infinity,
                            fit: BoxFit.cover,
                          ),
                        ),
                      ],

                      const SizedBox(height: 24),

                      // Timestamp info
                      Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: _bg,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: _border),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.access_time_rounded, size: 16, color: _textSec),
                            const SizedBox(width: 8),
                            Text(
                              "Thời gian chẩn đoán",
                              style: GoogleFonts.inter(fontSize: 12, color: _textSec),
                            ),
                            const Spacer(),
                            Text(
                              "${entry.timestamp.day.toString().padLeft(2, '0')}/${entry.timestamp.month.toString().padLeft(2, '0')}/${entry.timestamp.year}  "
                                  "${entry.timestamp.hour.toString().padLeft(2, '0')}:${entry.timestamp.minute.toString().padLeft(2, '0')}",
                              style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: _textPri),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 32),

                      // Close button
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: () => Navigator.pop(ctx),
                          style: FilledButton.styleFrom(
                            backgroundColor: _textPri,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                          ),
                          child: Text(
                            "Đóng",
                            style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 14),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 16, color: _textSec),
        const SizedBox(width: 8),
        Text(label, style: GoogleFonts.inter(fontSize: 12, color: _textSec)),
        const Spacer(),
        Expanded(
          child: Text(
            value,
            textAlign: TextAlign.end,
            style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: _textPri),
          ),
        ),
      ],
    );
  }
}

// ─── Sub-widgets ──────────────────────────────────────────────────────────────

class _DetailSectionLabel extends StatelessWidget {
  final String text;
  const _DetailSectionLabel(this.text);
  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w600, color: _textSec, letterSpacing: 0.2),
    );
  }
}

class _ProbBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  const _ProbBar({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(color: color, shape: BoxShape.circle),
                ),
                const SizedBox(width: 8),
                Text(label, style: GoogleFonts.inter(fontSize: 13, color: _textPri, fontWeight: FontWeight.w500)),
              ],
            ),
            Text(
              "${(value * 100).toStringAsFixed(1)}%",
              style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w700, color: _textPri),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value,
            minHeight: 7,
            backgroundColor: _border,
            color: color,
          ),
        ),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final int value;
  final IconData icon;
  final Color color;
  const _StatCard({required this.label, required this.value, required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 10),
      decoration: BoxDecoration(
        color: _white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _border),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.02), blurRadius: 6, offset: const Offset(0, 2)),
        ],
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              color: color.withOpacity(0.08),
              borderRadius: BorderRadius.circular(9),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(height: 10),
          Text(
            value.toString(),
            style: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.w800, color: _textPri),
          ),
          const SizedBox(height: 3),
          Text(
            label,
            style: GoogleFonts.inter(fontSize: 11, color: _textSec, fontWeight: FontWeight.w500),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}