package com.example.ailowcurrentengineer.config;

import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
public class MinioConfig {

    @Bean
    public MinioClient minioClient(
            @Value("${s3.endpoint}") String endpoint,
            @Value("${s3.access-key}") String accessKey,
            @Value("${s3.secret-key}") String secretKey
    ) {
        return MinioClient.builder()
                .endpoint(endpoint)
                .credentials(accessKey, secretKey)
                .build();
    }
}
