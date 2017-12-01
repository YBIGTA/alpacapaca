package com.ybigta.alpacapaca.autoreply.model;

import lombok.Data;

import java.io.Serializable;
import java.util.List;

@Data
public class AlpacapacaMessage implements Serializable {
    private boolean success;
    private List<String> results;
}
