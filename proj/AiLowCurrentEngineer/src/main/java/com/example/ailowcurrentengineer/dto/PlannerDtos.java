package com.example.ailowcurrentengineer.dto;

import java.util.List;
import java.util.Map;

public class PlannerDtos {

  // Python /ingest
  public record IngestRequest(String project_id, String src_s3_key) {}
  public record IngestResponse(String project_id, Integer rooms, String note) {}

  // Python /lighting (если понадобится дергать отдельно)
  public record LightingRequest(
      String project_id,
      int total_fixtures,
      Double target_lux,
      Map<String, Double> per_room_target_lux,
      Double fixture_efficacy_lm_per_w,
      Double maintenance_factor,
      Double utilization_factor
  ) {}

  // Python /infer — единая точка
  public record InferRequest(
      String project_id,
      String user_preferences_text,
      Integer default_total_fixtures,
      Double default_target_lux,
      Double fixture_efficacy_lm_per_w,
      Double maintenance_factor,
      Double utilization_factor,
      List<String> export_formats
  ) {}

  public record InferResponse(
      String project_id,
      Map<String, Object> parsed,
      Map<String, Object> lighting,
      List<String> exported_files,
      List<String> uploaded_uris
  ) {}

  // Наш UI
  public record CreateProjectResponse(String id, String name) {}
  public record UploadResponse(String projectId, String key) {}

  public record RunJobRequest(
      String projectId,
      String preferencesText,
      Integer totalFixtures,
      Double targetLux,
      Double efficacyLmPerW,
      Double maintenanceFactor,
      Double utilizationFactor,
      List<String> exportFormats
  ) {}

  public record JobResponse(
      String jobId,
      String status,
      Map<String, Object> result
  ) {}
}
