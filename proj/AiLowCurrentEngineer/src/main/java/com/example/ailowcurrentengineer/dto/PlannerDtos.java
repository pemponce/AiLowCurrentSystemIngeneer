package com.example.ailowcurrentengineer.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

public class PlannerDtos {

    // Python /lighting (если понадобится дергать отдельно)
    public record IngestRequest(
            @JsonProperty("projectId") String projectId,
            @JsonProperty("srcKey") String srcKey
    ) {
    }

    public record IngestResponse(
            @JsonProperty("project_id") String projectId,
            @JsonProperty("rooms") Integer rooms,
            @JsonProperty("note") String note
    ) {
    }

    public record InferRequest(
            @JsonProperty("projectId") String projectId,
            @JsonProperty("preferencesText") String preferencesText,
            @JsonProperty("totalFixtures") Integer totalFixtures,
            @JsonProperty("targetLux") Double targetLux,
            @JsonProperty("efficacyLmPerW") Double efficacyLmPerW,
            @JsonProperty("maintenanceFactor") Double maintenanceFactor,
            @JsonProperty("utilizationFactor") Double utilizationFactor,
            @JsonProperty("exportFormats") List<String> exportFormats
    ) {
    }

    public record InferResponse(
            @JsonProperty("project_id") String projectId,
            @JsonProperty("lighting") Object lighting,
            @JsonProperty("parsed") Object parsed,
            @JsonProperty("exported_files") List<String> exportedFiles,
            @JsonProperty("uploaded_uris") List<String> uploadedUris
    ) {
    }

    // Наш UI
    public record CreateProjectResponse(String id, String name) {
    }

    public record UploadResponse(String projectId, String key) {
    }

    // PlannerDtos.java (фрагмент)
    public record RunJobRequest(
            @com.fasterxml.jackson.annotation.JsonAlias({"project_id"}) String projectId,
            @com.fasterxml.jackson.annotation.JsonAlias({"src_key"}) String srcKey,
            @com.fasterxml.jackson.annotation.JsonAlias({"preferences_text"}) String preferencesText,
            @com.fasterxml.jackson.annotation.JsonAlias({"export_formats"}) java.util.List<String> exportFormats,
            Integer totalFixtures,
            Double targetLux,
            Double efficacyLmPerW,
            Double maintenanceFactor,
            Double utilizationFactor
    ) {}

    public record JobResponse(
            String jobId,
            String status,
            Map<String, Object> result
    ) {
    }
}
