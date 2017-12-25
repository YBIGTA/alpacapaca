package com.ybigta.alpacapaca.autoreply.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class MessageRequest {
    @JsonProperty("user_key")
    private String userKey;
    private String type;
    private String content;
}
