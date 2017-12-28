package com.ybigta.alpacapaca.autoreply.model;

import lombok.Data;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Data
@Entity
public class AlpacapacaRecord {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;
    private String userKey;
    private String input;
    private String output;
    private Long requestTime;
}
