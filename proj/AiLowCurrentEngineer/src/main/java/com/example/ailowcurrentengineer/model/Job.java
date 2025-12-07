package com.example.ailowcurrentengineer.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.UuidGenerator;

import java.time.OffsetDateTime;
import java.util.UUID;

@Data
@Entity
@Table(name = "jobs")
@AllArgsConstructor
@NoArgsConstructor

public class Job {
  public enum Status { PENDING, RUNNING, DONE, ERROR }

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @ManyToOne(optional = false, fetch = FetchType.LAZY)
  @JoinColumn(name = "project_id", nullable = false)
  private Project project;

  @Enumerated(EnumType.STRING)
  @Column(nullable = false)
  private Status status;

  @Column(name = "preferences_text")
  private String preferencesText;

  @Column(name = "total_fixtures")
  private Integer totalFixtures;

  @Column(name = "target_lux")
  private Double targetLux;

  @Column(name = "efficacy_lm_per_w")
  private Double efficacyLmPerW;

  @Column(name = "maintenance_factor")
  private Double maintenanceFactor;

  @Column(name = "utilization_factor")
  private Double utilizationFactor;

  @Column(name = "export_formats")
  private String exportFormats; // JSON string ["PDF","DXF"]

  @Column(name = "parsed_json")
  private String parsedJson;

  @Column(name = "lighting_json")
  private String lightingJson;

  @Column(name = "exported_files_json")
  private String exportedFilesJson;

  @Column(name = "uploaded_uris_json")
  private String uploadedUrisJson;

  @Column(name = "error_text")
  private String errorText;

  @Column(name = "created_at", nullable = false)
  private OffsetDateTime createdAt;

  @Column(name = "updated_at", nullable = false)
  private OffsetDateTime updatedAt;

  @PrePersist
  public void onCreate() {
    OffsetDateTime now = OffsetDateTime.now();
    createdAt = now;
    updatedAt = now;
  }

  @PreUpdate
  public void onUpdate() {
    updatedAt = OffsetDateTime.now();
  }
}
