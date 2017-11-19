package com.ybigta.alpacapaca.autoreply.model;

import lombok.Data;

@Data
public class Request {
    private String userKey;
    private String type;
    private String content;
}
