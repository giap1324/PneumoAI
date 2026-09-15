import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:marquee/marquee.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(
        backgroundColor: Colors.white.withValues(alpha: 0.5),
        surfaceTintColor: Colors.transparent,
        foregroundColor: const Color(0xFF212529),
        elevation: 0,
        centerTitle: false,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFEEEEF0)),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Logo container
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF00B4DB), Color(0xFF7C4DFF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(11),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF7C4DFF).withValues(alpha: 0.25),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Icon(
                Icons.info_outline_rounded,
                color: Colors.white,
                size: 20,
              ),
            ),

            const SizedBox(width: 12),

            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ShaderMask(
                  blendMode: BlendMode.srcIn,
                  shaderCallback: (bounds) => const LinearGradient(
                    colors: [Color(0xFF00B4DB), Color(0xFF7C4DFF)],
                    begin: Alignment.centerLeft,
                    end: Alignment.centerRight,
                  ).createShader(bounds),
                  child: Text(
                    "Giới thiệu hệ thống",
                    style: GoogleFonts.poppins(
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.2,
                      color: Colors.white,
                    ),
                  ),
                ),
                Text(
                  "Pneumonia Detection AI",
                  style: GoogleFonts.inter(
                    fontSize: 11,
                    fontWeight: FontWeight.w400,
                    color: const Color(0xFF9CA3AF),
                    letterSpacing: 0.1,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: const [
          SizedBox(width: 16),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 110),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 10),
            // Logo & Title
            Image.asset('lib/assets/logo.png', width: 80, height: 80),
            const SizedBox(height: 20),
            RichText(
              text: TextSpan(
                children: [
                  const TextSpan(
                    text: "Pneumo",
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                      letterSpacing: 0.5,
                    ),
                  ),

                  WidgetSpan(
                    alignment: PlaceholderAlignment.middle,
                    child: ShaderMask(
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
                        "AI",
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              "Hệ thống hỗ trợ chẩn đoán viêm phổi qua X-quang",
              textAlign: TextAlign.center,
              style: TextStyle(color: Color(0xFF8B949E), fontSize: 14),
            ),
            
            const SizedBox(height: 40),
            
            // Stats Row
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildStat("97.95%", "Accuracy"),
                _buildStat("<200ms", "Latency"),
              ],
            ),
            
            const SizedBox(height: 40),
            
            _buildSectionTitle("Về mô hình AI"),
            const SizedBox(height: 16),
            _buildInfoCard(
              "Mô hình sử dụng kiến trúc Hybrid Vision Transformer (ViT) kết hợp với DenseNet121. Sự kết hợp này cho phép hệ thống vừa trích xuất đặc trưng cục bộ tốt (CNN), vừa hiểu được mối quan hệ toàn cục trong ảnh (Transformer).",
            ),
            
            const SizedBox(height: 32),
            
            _buildSectionTitle("Công nghệ sử dụng"),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _buildTechChip("Flutter"),
                _buildTechChip("PyTorch"),
                _buildTechChip("FastAPI"),
                _buildTechChip("MySQL"),
                _buildTechChip("Vision Transformer"),
              ],
            ),
            
            const SizedBox(height: 40),

            _buildSectionTitle("Thông tin liên hệ"),

            const SizedBox(height: 16),

            const ListTile(
              contentPadding: EdgeInsets.zero,

              leading: Icon(
                Icons.email_outlined,
                color: Color(0xFF2563FF),
              ),

              title: Text(
                "Email hỗ trợ",
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF64748B),
                ),
              ),

              subtitle: Text(
                "support@pneumoscan.ai",
                style: TextStyle(
                  color: Color(0xFF1E293B),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            const ListTile(
              contentPadding: EdgeInsets.zero,

              leading: Icon(
                Icons.language_outlined,
                color: Color(0xFF2563FF),
              ),

              title: Text(
                "Website",
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF64748B),
                ),
              ),

              subtitle: Text(
                "www.pneumo.ai",
                style: TextStyle(
                  color: Color(0xFF1E293B),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            const ListTile(
              contentPadding: EdgeInsets.zero,

              leading: Icon(
                Icons.code_rounded,
                color: Color(0xFF2563FF),
              ),

              title: Text(
                "GitHub",
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF64748B),
                ),
              ),

              subtitle: Text(
                "github.com/pneumoai",
                style: TextStyle(
                  color: Color(0xFF1E293B),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            ListTile(
              contentPadding: EdgeInsets.zero,

              leading: Icon(
                Icons.school_outlined,
                color: Color(0xFF2563FF),
              ),

              title: Text(
                "Đơn vị phát triển",
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF64748B),
                ),
              ),

              subtitle: SizedBox(
                height: 22,

                child: Marquee(
                  text:
                  "Sinh viên Công nghệ Thông tin • Trường Đại học Đại Nam",

                  style: const TextStyle(
                    color: Color(0xFF1E293B),
                    fontWeight: FontWeight.w600,
                    height: 1.4,
                  ),

                  velocity: 28,
                  blankSpace: 40,
                  pauseAfterRound: const Duration(seconds: 1),
                  startPadding: 10,
                  accelerationDuration: const Duration(milliseconds: 500),
                  decelerationDuration: const Duration(milliseconds: 500),
                ),
              ),
            ),
            
            const SizedBox(height: 40),
            const Divider(color: Color(0xFFE9ECEF)),
            const SizedBox(height: 20),
            const Text(
              "© 2026 Pneumo AI \nỨng dụng phục vụ mục đích nghiên cứu học thuật.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Color(0xFF8B949E), fontSize: 11, height: 1.5),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildStat(String val, String label) {
    return Column(
      children: [
        Text(val, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Color(0xFF212529))),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 12, color: Color(0xFF6C757D))),
      ],
    );
  }

  Widget _buildSectionTitle(String title) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(
        title.toUpperCase(),
        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFF0D6EFD), letterSpacing: 1),
      ),
    );
  }

  Widget _buildInfoCard(String text) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F9FA),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE9ECEF)),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Color(0xFF212529), fontSize: 14, height: 1.6),
      ),
    );
  }

  Widget _buildTechChip(String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFFE9ECEF),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFDEE2E6)),
      ),
      child: Text(label, style: const TextStyle(fontSize: 12, color: Color(0xFF212529))),
    );
  }
}
