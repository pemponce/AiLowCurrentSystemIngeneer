package com.example.ailowcurrentengineer.service;

import com.example.ailowcurrentengineer.model.Job;
import com.example.ailowcurrentengineer.model.Project;
import com.example.ailowcurrentengineer.dto.PlannerDtos.InferResponse;
import com.example.ailowcurrentengineer.dto.PlannerDtos.RunJobRequest;
import com.example.ailowcurrentengineer.repository.JobRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
public class JobService {
  private final JobRepository jobs;
  private final ProjectService projects;
  private final PlannerClient planner;
  private final ObjectMapper om = new ObjectMapper();

  public JobService(JobRepository jobs, ProjectService projects, PlannerClient planner) {
    this.jobs = jobs;
    this.projects = projects;
    this.planner = planner;
  }

  @Transactional
  public Job createPending(Long projectId, RunJobRequest req) {
    Project p = projects.get(projectId);

    Job j = new Job();
    j.setProject(p);
    j.setStatus(Job.Status.PENDING);
    j.setPreferencesText(req.preferencesText());
    j.setTotalFixtures(req.totalFixtures());
    j.setTargetLux(req.targetLux());
    j.setEfficacyLmPerW(req.efficacyLmPerW());
    j.setMaintenanceFactor(req.maintenanceFactor());
    j.setUtilizationFactor(req.utilizationFactor());
    try {
      String formatsJson = om.writeValueAsString(req.exportFormats() != null ? req.exportFormats() : List.of("PDF","DXF"));
      j.setExportFormats(formatsJson);
    } catch (Exception ignored) {}

    return jobs.save(j);
  }

  @Transactional
  public Job runNow(Job j, RunJobRequest req) {
    j.setStatus(Job.Status.RUNNING);
    jobs.save(j);

    try {
      InferResponse infer = planner.infer(req);

      j.setStatus(Job.Status.DONE);
      j.setParsedJson(writeJson(infer.parsed()));
      j.setLightingJson(writeJson(infer.lighting()));
      j.setExportedFilesJson(writeJson(infer.exportedFiles()));
      j.setUploadedUrisJson(writeJson(infer.uploadedUris()));
    } catch (Exception e) {
      j.setStatus(Job.Status.ERROR);
      j.setErrorText(e.getMessage());
    }
    return jobs.save(j);
  }

  public Job get(Long id) {
    return jobs.findById(id).orElseThrow();
  }

  private String writeJson(Object o) {
    try { return om.writeValueAsString(o); }
    catch (Exception e) { return null; }
  }
}
