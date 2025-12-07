package com.example.ailowcurrentengineer.repository;

import com.example.ailowcurrentengineer.model.Project;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface ProjectRepository extends JpaRepository<Project, Long> {}
