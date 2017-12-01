package com.ybigta.alpacapaca.autoreply.service.validator;

import lombok.Data;

@Data
public class ValidationResult {
    private boolean valid;
    private String message;
}
