package com.example.ailowcurrentengineer.service;

import com.example.ailowcurrentengineer.model.Project;
import com.example.ailowcurrentengineer.repository.ProjectRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


@Service
public class ProjectService {
    private final ProjectRepository projects;

    public ProjectService(ProjectRepository projects) {
        this.projects = projects;
    }

    @Transactional
    public Project create(String name) {
        Project p = new Project();
        p.setName(name);
        return projects.save(p);
    }

    @Transactional
    public Project setSrcKey(Long id, String srcKey) {
        Project p = projects.findById(id).orElseThrow();
        p.setSrcKey(srcKey);
        return projects.save(p);
    }

    public Project findById(Long id) {
        return projects.findById(id).orElseThrow();
    }

    public Project get(Long id) {
        return projects.findById(id).orElseThrow();
    }
}
