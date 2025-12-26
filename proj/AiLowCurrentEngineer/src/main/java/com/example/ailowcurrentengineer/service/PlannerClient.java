package com.example.ailowcurrentengineer.service;

import com.example.ailowcurrentengineer.dto.PlannerDtos.*;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.*;

@Service
public class PlannerClient {

    private final WebClient wc;

    public PlannerClient(@Qualifier("plannerWebClient") WebClient plannerWebClient) {
        this.wc = plannerWebClient;
    }

    /**
     * /ingest — ТОЛЬКО camelCase ключи
     */
    public Map<String, Object> ingest(String projectId, String srcKey) {
        Map<String, Object> payload = Map.of(
                "projectId", String.valueOf(projectId), // planner явно хочет строку
                "srcKey", srcKey
        );

        return wc.post()
                .uri("/ingest")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .onStatus(s -> s.is4xxClientError() || s.is5xxServerError(),
                        r -> r.bodyToMono(String.class).map(body ->
                                new RuntimeException("Planner /ingest " + r.statusCode() + ": " + body)))
                .bodyToMono(Map.class)
                .block();
    }

    /**
     * /infer — camelCase и валидные exportFormats
     */
    public InferResponse infer(RunJobRequest runReq) {
        List<String> exportFormats = (runReq.exportFormats() != null && !runReq.exportFormats().isEmpty())
                ? runReq.exportFormats().stream().map(s -> s.toUpperCase()).toList()
                : List.of("PDF", "DXF");

        Map<String, Object> payload = new HashMap<>();
        payload.put("projectId", String.valueOf(runReq.projectId()));
        // Чтобы не ловить "Нет геометрии помещений", либо до этого зови /ingest, либо передай srcKey сюда:
        if (runReq.srcKey() != null && !runReq.srcKey().isBlank()) {
            payload.put("srcKey", runReq.srcKey());
        }
        if (runReq.preferencesText() != null) payload.put("preferencesText", runReq.preferencesText());
        payload.put("totalFixtures", runReq.totalFixtures() != null ? runReq.totalFixtures() : 20);
        payload.put("targetLux", runReq.targetLux() != null ? runReq.targetLux() : 300.0);
        payload.put("efficacyLmPerW", runReq.efficacyLmPerW() != null ? runReq.efficacyLmPerW() : 110.0);
        payload.put("maintenanceFactor", runReq.maintenanceFactor() != null ? runReq.maintenanceFactor() : 0.8);
        payload.put("utilizationFactor", runReq.utilizationFactor() != null ? runReq.utilizationFactor() : 0.6);
        payload.put("exportFormats", exportFormats); // ДОЛЖНЫ быть "PDF" | "DXF" | "PNG" в верхнем регистре

        return wc.post()
                .uri("/infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .onStatus(s -> s.is4xxClientError() || s.is5xxServerError(),
                        r -> r.bodyToMono(String.class).map(body ->
                                new RuntimeException("Planner /infer " + r.statusCode() + ": " + body)))
                .bodyToMono(InferResponse.class)
                .block();
    }
}
