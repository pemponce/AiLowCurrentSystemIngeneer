package com.example.ailowcurrentengineer.repository;

import com.example.ailowcurrentengineer.model.Job;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;
@Repository
public interface JobRepository extends JpaRepository<Job, Long> {}
