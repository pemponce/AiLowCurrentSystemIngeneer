package com.example.ailowcurrentengineer.model;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.UuidGenerator;

import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

@Data
@Table(name = "users")
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Entity
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id", nullable = false, updatable = false)
    private Long id;
    @Column(name = "email", nullable = false, length = 255)
    private String email;

    @Column(name = "full_name", length = 255)
    private String fullName;

    @Column(name = "password", nullable = false, length = 255)
    private String password;

    @Builder.Default
    @Column(name = "enabled", nullable = false)
    private boolean enabled = true;

    @Builder.Default
    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(
            name = "user_roles",
            joinColumns = @JoinColumn(name = "user_id", referencedColumnName = "id"),
            inverseJoinColumns = @JoinColumn(name = "role_id", referencedColumnName = "id")
    )
    private Set<Role> roles = new HashSet<>();
}
