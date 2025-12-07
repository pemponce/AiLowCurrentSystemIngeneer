package com.example.ailowcurrentengineer.service;

import com.example.ailowcurrentengineer.dto.PlannerDtos.*;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.Map;

@Service
public class PlannerClient {

  private final WebClient wc;

  public PlannerClient(WebClient plannerWebClient) {
    this.wc = plannerWebClient;
  }

  public IngestResponse ingest(String projectId, String srcKey) {
    IngestRequest req = new IngestRequest(projectId, srcKey);
    return wc.post()
        .uri("/ingest")
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(req)
        .retrieve()
        .bodyToMono(IngestResponse.class)
        .block();
  }

  public InferResponse infer(RunJobRequest runReq) {
    // Собираем payload с дефолтами
    InferRequest req = new InferRequest(
        runReq.projectId(),
        runReq.preferencesText(),
        runReq.totalFixtures() != null ? runReq.totalFixtures() : 20,
        runReq.targetLux() != null ? runReq.targetLux() : 300.0,
        runReq.efficacyLmPerW() != null ? runReq.efficacyLmPerW() : 110.0,
        runReq.maintenanceFactor() != null ? runReq.maintenanceFactor() : 0.8,
        runReq.utilizationFactor() != null ? runReq.utilizationFactor() : 0.6,
        runReq.exportFormats() != null ? runReq.exportFormats() : List.of("PDF","DXF")
    );

    return wc.post()
        .uri("/infer")
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(req)
        .retrieve()
        .bodyToMono(InferResponse.class)
        .block();
  }
}
