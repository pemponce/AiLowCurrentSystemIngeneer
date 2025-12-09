package com.example.ailowcurrentengineer.service;

import io.minio.BucketExistsArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.UUID;

@Service
public class S3Service {

  private final MinioClient minio;
  private final String bucketRaw;
  private final String bucketExports;

  public S3Service(
          MinioClient minio,
          @Value("${s3.bucket-raw}") String bucketRaw,
          @Value("${s3.bucket-exports}") String bucketExports  ) {
    this.minio = minio;
    this.bucketRaw = bucketRaw;
    this.bucketExports = bucketExports;
  }

    public String uploadRawBytes(String objectKey, byte[] data, String contentType) throws Exception {
        ensureBucket(bucketRaw);
        try (var in = new java.io.ByteArrayInputStream(data)) {
            PutObjectArgs args = PutObjectArgs.builder()
                    .bucket(bucketRaw)
                    .object(objectKey)
                    .stream(in, data.length, -1)
                    .contentType(contentType != null ? contentType : "application/octet-stream")
                    .build();
            minio.putObject(args);
        }
        return objectKey;
    }


    public String uploadRawPlan(String projectId, MultipartFile file) throws Exception {
    ensureBucket(bucketRaw);

    String key = "raw-plans/" + projectId + "/" + UUID.randomUUID() + "-" + sanitize(file.getOriginalFilename());
    try (InputStream is = file.getInputStream()) {
      PutObjectArgs args = PutObjectArgs.builder()
              .bucket(bucketRaw)
              .object(key)
              .stream(is, file.getSize(), -1)
              .contentType(file.getContentType() != null ? file.getContentType() : "application/octet-stream")
              .build();
      minio.putObject(args);
    }
    return key;
  }

  public String uploadExport(String objectKey, InputStream content, long size, String contentType) throws Exception {
    ensureBucket(bucketExports);
    PutObjectArgs args = PutObjectArgs.builder()
            .bucket(bucketExports)
            .object(objectKey)
            .stream(content, size, -1)
            .contentType(contentType != null ? contentType : "application/octet-stream")
            .build();
    minio.putObject(args);
    return objectKey;
  }

  private void ensureBucket(String bucket) throws Exception {
    boolean exists = minio.bucketExists(BucketExistsArgs.builder().bucket(bucket).build());
    if (!exists) {
      minio.makeBucket(MakeBucketArgs.builder().bucket(bucket).build());
    }
  }

  private static String sanitize(String name) {
    if (name == null) return "file.bin";
    return name.replaceAll("[^a-zA-Z0-9._-]", "_");
  }
}
