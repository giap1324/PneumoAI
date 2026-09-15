import 'package:flutter/material.dart';

// ─── Constants ──────────────────────────────────────────────────────────────
const kBg         = Colors.white;
const kSurface    = Color(0xFFF8F9FA);
const kCard       = Colors.white;
const kBorder     = Color(0xFFE9ECEF);
const kCyan       = Color(0xFF0D6EFD); // Blue primary
const kGreen      = Color(0xFF198754);
const kRed        = Color(0xFFDC3545);
const kTextPrim   = Color(0xFF212529);
const kTextSecond = Color(0xFF6C757D);

// ─── Model Result ───────────────────────────────────────────────────────────
class PredictionResult {
  final String label;
  final double confidence;
  final double normalProb;
  final double pneumoniaProb;
  final String? imageUrl;
  final String? heatmapUrl;
  final String? gradcamUrl;
  final SeverityData? severity;
  final double? inferenceTime;
  final double? gradcamTime;
  final double? preprocessTime;

  PredictionResult({
    required this.label,
    required this.confidence,
    required this.normalProb,
    required this.pneumoniaProb,
    this.imageUrl,
    this.heatmapUrl,
    this.gradcamUrl,
    this.severity,
    this.inferenceTime,
    this.gradcamTime,
    this.preprocessTime,
  });

  bool get isPneumonia => label == "PNEUMONIA";
}

class SeverityData {
  final double compositeScore;
  final double areaRatio;
  final double meanIntensity;
  final double affectedAreaPct;
  final double leftAreaPct;
  final double rightAreaPct;
  final int brixiaScore;
  final List<int> zoneScores;
  final String gradeEn;
  final String gradeVi;
  final String gradeDescription;

  SeverityData({
    required this.compositeScore,
    required this.areaRatio,
    required this.meanIntensity,
    required this.affectedAreaPct,
    required this.leftAreaPct,
    required this.rightAreaPct,
    required this.brixiaScore,
    required this.zoneScores,
    required this.gradeEn,
    required this.gradeVi,
    required this.gradeDescription,
  });

  factory SeverityData.fromJson(Map<String, dynamic> json) {
    return SeverityData(
      compositeScore: (json['composite_score'] ?? 0.0) as double,
      areaRatio: (json['area_ratio'] ?? 0.0) as double,
      meanIntensity: (json['mean_intensity'] ?? 0.0) as double,
      affectedAreaPct: (json['affected_area_pct'] ?? 0.0) as double,
      leftAreaPct: (json['left_area_pct'] ?? 0.0) as double,
      rightAreaPct: (json['right_area_pct'] ?? 0.0) as double,
      brixiaScore: json['brixia_score'] ?? 0,
      zoneScores: (json['zone_scores'] as List<dynamic>?)?.map((e) => e as int).toList() ?? [],
      gradeEn: json['grade_en'] ?? 'N/A',
      gradeVi: json['grade_vi'] ?? 'N/A',
      gradeDescription: json['grade_description'] ?? '',
    );
  }
}
