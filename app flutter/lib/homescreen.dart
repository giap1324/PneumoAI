import 'dart:convert';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'shared.dart';
import 'resultscreen.dart';
import 'historyscreen.dart';
import 'package:google_fonts/google_fonts.dart';

// ── Color tokens ─────────────────────────────────────────────────────────────
const kTeal       = Color(0xFF1D9E75);
const kTealLight   = Color(0xFFE1F5EE);
const kTealMid    = Color(0xFF5DCAA5);
const kTealDark   = Color(0xFF0F6E56);
const kBgPage     = Color(0xFFF7F7F5);
const kCard       = Color(0xFFFFFFFF);
const kBorder     = Color(0xFFE0E0E0);
const kBgSecond   = Color(0xFFF1F0EA);
const kTextPrimary = Color(0xFF2C2C2A);
const kTextSecond  = Color(0xFF888780);
const kRed        = Color(0xFFE24B4A);
const kRedLight   = Color(0xFFFCEBEB);

// ─────────────────────────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _selectedImage;
  bool  _loading  = false;
  String? _errorMsg;

  final _nameController = TextEditingController();
  final _ageController  = TextEditingController();
  final _genderController = TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _genderController.dispose();
    super.dispose();
  }

  final String baseUrl = "http://10.0.2.2:8000";

  // ── Actions ────────────────────────────────────────────────────────────────

  Future<void> _pickImage() async {
    final picked = await FilePicker.platform.pickFiles(type: FileType.image);
    if (picked != null) {
      setState(() {
        _selectedImage = File(picked.files.single.path!);
        _errorMsg = null;
      });
    }
  }

  Future<void> _predict() async {
    if (_selectedImage == null) return;
    setState(() { _loading = true; _errorMsg = null; });

    try {
      final uri     = Uri.parse("$baseUrl/predict");
      final request = http.MultipartRequest("POST", uri);
      request.files.add(
        await http.MultipartFile.fromPath("file", _selectedImage!.path),
      );
      request.fields['patient_name'] = _nameController.text;
      request.fields['patient_age'] = _ageController.text;
      request.fields['gender'] = _genderController.text;

      final response     = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (!mounted) return;

      if (response.statusCode == 200) {
        final data   = jsonDecode(responseBody);
        final result = PredictionResult(
          label:         data["label"],
          confidence:    (data["confidence"] as num).toDouble(),
          normalProb:    (data["probabilities"]["NORMAL"] as num).toDouble(),
          pneumoniaProb: (data["probabilities"]["PNEUMONIA"] as num).toDouble(),
          imageUrl:      data["image_url"] != null
              ? baseUrl + data["image_url"]
              : null,
          heatmapUrl:    data["heatmap_url"] != null
              ? baseUrl + data["heatmap_url"]
              : null,
          gradcamUrl:    data["gradcam_url"] != null
              ? baseUrl + data["gradcam_url"]
              : null,
          severity: data["severity"] != null
              ? SeverityData.fromJson(data["severity"])
              : null,
          inferenceTime: data["inference_time"] != null
              ? (data["inference_time"] as num).toDouble()
              : null,
          gradcamTime: data["gradcam_time"] != null
              ? (data["gradcam_time"] as num).toDouble()
              : null,
          preprocessTime: data["preprocess_time"] != null
              ? (data["preprocess_time"] as num).toDouble()
              : null,
        );

        HistoryStore().fetchHistory();
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              result: result,
              originalImage: _selectedImage!,
              patientName: _nameController.text.isNotEmpty ? _nameController.text : null,
              patientAge: _ageController.text.isNotEmpty ? _ageController.text : null,
              gender: _genderController.text.isNotEmpty ? _genderController.text : null,
            ),
          ),
        );
      } else {
        setState(() { _errorMsg = "Lỗi từ server: ${response.statusCode}"; });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() { _errorMsg = "Lỗi kết nối: $e"; });
    } finally {
      if (mounted) setState(() { _loading = false; });
    }
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: _buildAppBar(),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 24, 20, 110),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildPageHeader(),
              const SizedBox(height: 32),
              _buildUploadCard(),
              if (_errorMsg != null) ...[
                const SizedBox(height: 16),
                _buildErrorBanner(),
              ],
              if (_loading) ...[
                const SizedBox(height: 32),
                _buildLoadingState(),
              ],
              const SizedBox(height: 32),
              _buildTechSection(),
              const SizedBox(height: 24),
              _buildDisclaimer(),
            ],
          ),
        ),
      ),
    );
  }

  // ── AppBar ─────────────────────────────────────────────────────────────────

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      backgroundColor: Colors.white.withValues(alpha: 0.5),
      elevation: 0,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 16,
      title: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: kTealLight,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Padding(
              padding: const EdgeInsets.all(5),
              child: Image.asset(
                'lib/assets/logo.png',
                fit: BoxFit.contain,
              ),
            ),
          ),

          const SizedBox(width: 12),

          RichText(
            text: TextSpan(
              children: [
                TextSpan(
                  text: "Pneumo",
                  style: GoogleFonts.poppins(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    color: Colors.black,
                    letterSpacing: 1,
                  ),
                ),

                WidgetSpan(
                  alignment: PlaceholderAlignment.middle,
                  child: ShaderMask(
                    blendMode: BlendMode.srcIn,
                    shaderCallback: (bounds) {
                      return const LinearGradient(
                        colors: [
                          Color(0xFF00E5FF),
                          Color(0xFF7C4DFF),
                        ],
                      ).createShader(bounds);
                    },
                    child: Text(
                      "AI",
                      style: GoogleFonts.poppins(
                        fontSize: 24,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        letterSpacing: 1,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),

      actions: [
        Container(
          margin: const EdgeInsets.only(right: 16),

          padding: const EdgeInsets.symmetric(
            horizontal: 14,
            vertical: 7,
          ),

          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [
                Color(0xFF00B4FF),
                Color(0xFF2563FF),
              ],
            ),

            borderRadius: BorderRadius.circular(99),

            boxShadow: [
              BoxShadow(
                color: const Color(0xFF2563FF).withValues(alpha: 0.25),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),

          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.auto_awesome,
                size: 13,
                color: Colors.white,
              ),

              SizedBox(width: 5),

              Text(
                'AI Phân tích',
                style: TextStyle(
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        ),
      ],

      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.5),
        child: Container(
          height: 0.5,
          color: kBorder,
        ),
      ),
    );
  }

  // ── Page header ────────────────────────────────────────────────────────────

  Widget _buildPageHeader() {
    return Column(
      children: [
        const SizedBox(height: 20),
        Text(
          'Phát Hiện Viêm Phổi Qua',
          style: GoogleFonts.poppins(
            fontSize: 28,
            fontWeight: FontWeight.w800,
            color: const Color(0xFF1E293B),
            height: 1.2,
          ),
          textAlign: TextAlign.center,
        ),
        ShaderMask(
          blendMode: BlendMode.srcIn,
          shaderCallback: (bounds) {
            return const LinearGradient(
              colors: [
                Color(0xFF7C3AED),
                Color(0xFF3B82F6),
              ],
            ).createShader(bounds);
          },
          child: Text(
            'X-Quang Ngực',
            style: GoogleFonts.poppins(
              fontSize: 32,
              fontWeight: FontWeight.w800,
              color: Colors.white,
            ),
          ),
        ),
        const SizedBox(height: 16),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 20),
          child: Text(
            'Ứng dụng công nghệ Hybrid ViT tiên tiến, cung cấp chẩn đoán nhanh chóng, chính xác và đánh giá mức độ nghiêm trọng.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: Color(0xFF64748B),
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }

  // ── Upload card ────────────────────────────────────────────────────────────

  Widget _buildUploadCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Phân Tích Chẩn Đoán',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1E293B),
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Định dạng hỗ trợ: JPG, PNG, DICOM',
            style: TextStyle(
              fontSize: 13,
              color: Color(0xFF94A3B8),
            ),
          ),
          const SizedBox(height: 24),
          GestureDetector(
            onTap: _loading ? null : _pickImage,
            child: CustomPaint(
              painter: DashRectPainter(color: const Color(0xFF7C3AED).withValues(alpha: 0.3)),
              child: Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  color: const Color(0xFFF8FAFC),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: _selectedImage == null
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: const BoxDecoration(
                              color: Color(0xFF7C3AED),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.cloud_upload_rounded, color: Colors.white, size: 28),
                          ),
                          const SizedBox(height: 16),
                          const Text(
                            'Kéo thả ảnh X-Quang vào đây',
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF1E293B),
                            ),
                          ),
                          const SizedBox(height: 4),
                          const Text(
                            'Hỗ trợ định dạng JPG, JPEG, PNG (tối đa 16MB)',
                            style: TextStyle(
                              fontSize: 12,
                              color: Color(0xFF94A3B8),
                            ),
                          ),
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: const Color(0xFF3B82F6)),
                            ),
                            child: const Text(
                              'Chọn Tệp Từ Máy Tính',
                              style: TextStyle(
                                color: Color(0xFF3B82F6),
                                fontWeight: FontWeight.w600,
                                fontSize: 14,
                              ),
                            ),
                          ),
                        ],
                      )
                    : Stack(
                        children: [
                          Positioned.fill(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(20),
                              child: Image.file(_selectedImage!, fit: BoxFit.cover),
                            ),
                          ),
                          // Nút xóa ảnh
                          Positioned(
                            top: 12,
                            right: 12,
                            child: GestureDetector(
                              onTap: () {
                                setState(() {
                                  _selectedImage = null;
                                  _errorMsg = null;
                                });
                              },
                              child: Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: Colors.black.withOpacity(0.5),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.close_rounded,
                                  color: Colors.white,
                                  size: 20,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                flex: 2,
                child: _buildInputBox(_nameController, 'Tên bệnh nhân', Icons.person_outline_rounded),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 1,
                child: _buildInputBox(_ageController, 'Tuổi', Icons.calendar_today_rounded, keyboardType: TextInputType.number),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _buildInputBox(_genderController, 'Giới tính (tùy chọn)', Icons.wc_rounded),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: (_selectedImage != null && !_loading) ? _predict : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF5D99F6),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                elevation: 0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  Icon(Icons.search, size: 20),
                  SizedBox(width: 8),
                  Text(
                    'Phân Tích Ảnh X-Quang',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Input Box Helper ──────────────────────────────────────────────────────

  Widget _buildInputBox(TextEditingController controller, String hint, IconData icon, {TextInputType? keyboardType}) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFF8FAFC),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        children: [
          Icon(icon, size: 18, color: const Color(0xFF94A3B8)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: controller,
              keyboardType: keyboardType,
              decoration: InputDecoration(
                border: InputBorder.none,
                hintText: hint,
                hintStyle: const TextStyle(color: Color(0xFF94A3B8), fontSize: 13),
                contentPadding: const EdgeInsets.symmetric(vertical: 12),
              ),
              style: const TextStyle(fontSize: 13, color: Color(0xFF1E293B)),
            ),
          ),
        ],
      ),
    );
  }

  // ── Tech Section ───────────────────────────────────────────────────────────

  Widget _buildTechSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            ShaderMask(
              blendMode: BlendMode.srcIn,
              shaderCallback: (bounds) {
                return const LinearGradient(
                  colors: [
                    Color(0xFF00B4FF),
                    Color(0xFF7C4DFF),
                  ],
                ).createShader(bounds);
              },

              child: const Text(
                'Công nghệ cốt lõi',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  letterSpacing: 0.3,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 18),
        _buildTechCard(
          icon: Icons.memory_rounded,
          title: 'Hybrid ViT',
          desc: 'Kết hợp CNN và Vision Transformer giúp tăng độ chính xác trong phân tích X-quang.',
          gradient: const [Color(0xFFEDE9FE), Color(0xFFF5F3FF)],
          iconColor: const Color(0xFF7C3AED),
        ),
        const SizedBox(height: 14),
        _buildTechCard(
          icon: Icons.layers_rounded,
          title: 'Grad-CAM',
          desc: 'Trực quan hóa vùng tổn thương giúp giải thích quyết định của mô hình AI.',
          gradient: const [Color(0xFFE0F2FE), Color(0xFFF0F9FF)],
          iconColor: const Color(0xFF0284C7),
        ),
      ],
    );
  }

  Widget _buildTechCard({
    required IconData icon,
    required String title,
    required String desc,
    required List<Color> gradient,
    required Color iconColor,
  }) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFF1F5F9)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: gradient),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: iconColor, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF1E293B),
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  desc,
                  style: const TextStyle(
                    fontSize: 12.5,
                    height: 1.5,
                    color: Color(0xFF64748B),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Loading state ──────────────────────────────────────────────────────────

  Widget _buildLoadingState() {
    return Column(
      children: [
        const SizedBox(
          width: 36,
          height: 36,
          child: CircularProgressIndicator(
            strokeWidth: 2.5,
            color: kTeal,
          ),
        ),
        const SizedBox(height: 14),
        const Text(
          'AI đang xử lý dữ liệu...',
          style: TextStyle(color: kTextSecond, fontSize: 13),
        ),
      ],
    );
  }

  // ── Error banner ───────────────────────────────────────────────────────────

  Widget _buildErrorBanner() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: kRedLight,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: kRed.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: kRed, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _errorMsg!,
              style: const TextStyle(color: kRed, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  // ── Disclaimer ─────────────────────────────────────────────────────────────

  Widget _buildDisclaimer() {
    return const Text(
      'Kết quả chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ để chẩn đoán chính xác.',
      textAlign: TextAlign.center,
      style: TextStyle(fontSize: 11, color: kTextSecond, height: 1.5),
    );
  }
}

class DashRectPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final double gap;

  DashRectPainter({this.color = Colors.black, this.strokeWidth = 1.0, this.gap = 5.0});

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    Path path = Path();
    path.addRRect(RRect.fromLTRBR(0, 0, size.width, size.height, const Radius.circular(20)));

    Path dashPath = Path();
    double dashWidth = 10.0;
    double dashSpace = 5.0;
    double distance = 0.0;

    for (var pathMetric in path.computeMetrics()) {
      while (distance < pathMetric.length) {
        dashPath.addPath(
          pathMetric.extractPath(distance, distance + dashWidth),
          Offset.zero,
        );
        distance += dashWidth + dashSpace;
      }
    }
    canvas.drawPath(dashPath, paint);
  }

  @override
  bool shouldRepaint(DashRectPainter oldDelegate) => false;
}