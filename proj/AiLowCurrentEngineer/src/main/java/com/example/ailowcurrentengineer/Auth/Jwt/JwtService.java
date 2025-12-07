// com/example/ailowcurrentengineer/service/JwtService.java
package com.example.ailowcurrentengineer.Auth.Jwt;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.security.Key;
import java.time.Instant;
import java.util.Date;
import java.util.Map;

@Service
public class JwtService {

    private final Key key;
    private final long ttlMillis;

    public JwtService(
            @Value("${security.jwt.secret}") String secretBase64,
            @Value("${security.jwt.ttlMillis}") long ttlMillis
    ) {
        this.key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(secretBase64));
        this.ttlMillis = ttlMillis;
    }

    public String generate(String subject, Map<String, Object> claims) {
        Instant now = Instant.now();
        return Jwts.builder()
                .subject(subject)
                .claims(claims)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plusMillis(ttlMillis)))
                .signWith(key)
                .compact();
    }
}
