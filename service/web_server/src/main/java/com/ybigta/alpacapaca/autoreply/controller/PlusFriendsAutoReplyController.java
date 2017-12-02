package com.ybigta.alpacapaca.autoreply.controller;

import com.ybigta.alpacapaca.autoreply.PayloadFieldTypes;
import com.ybigta.alpacapaca.autoreply.model.MessageRequest;
import com.ybigta.alpacapaca.autoreply.service.MessageGenerator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
public class PlusFriendsAutoReplyController {
    private final MessageGenerator messageGenerator;

    @Autowired
    public PlusFriendsAutoReplyController(final MessageGenerator messageGenerator) {
        this.messageGenerator = messageGenerator;
    }

    @GetMapping(value = "/keyboard")
    @ResponseStatus(HttpStatus.OK)
    public Map<String, Object> submitKeyboard() {
        Map<String, Object> response = new HashMap<>();
        response.put(PayloadFieldTypes.TYPE, PayloadFieldTypes.TEXT);

        return response;
    }

    @PostMapping(value = "/message")
    @ResponseStatus(HttpStatus.OK)
    public Map<String, Object> submitMessage(@RequestBody MessageRequest messageRequest) {
        Map<String, Object> response = new HashMap<>();
        Map<String, Object> message = new HashMap<>();

        String inputContent = messageRequest.getContent();
        String returnMessage = messageGenerator.generateMessage(inputContent);

        message.put(PayloadFieldTypes.TEXT, returnMessage);
        response.put(PayloadFieldTypes.MESSAGE, message);

        return response;
    }
}
