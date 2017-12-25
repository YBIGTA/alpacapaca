package com.ybigta.alpacapaca.autoreply.controller;

import com.ybigta.alpacapaca.autoreply.PayloadFieldTypes;
import com.ybigta.alpacapaca.autoreply.dao.AlpacapacaRecordRepository;
import com.ybigta.alpacapaca.autoreply.model.AlpacapacaRecord;
import com.ybigta.alpacapaca.autoreply.model.MessageRequest;
import com.ybigta.alpacapaca.autoreply.service.MessageGenerationResult;
import com.ybigta.alpacapaca.autoreply.service.MessageGenerator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

@RestController
public class PlusFriendsAutoReplyController {
    private final MessageGenerator messageGenerator;
    private final AlpacapacaRecordRepository recordRepository;

    @Autowired
    public PlusFriendsAutoReplyController(final MessageGenerator messageGenerator,
                                          final AlpacapacaRecordRepository repository) {
        this.messageGenerator = messageGenerator;
        this.recordRepository = repository;
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
        long serverTime = Instant.now().toEpochMilli();

        Map<String, Object> response = new HashMap<>();
        Map<String, Object> message = new HashMap<>();

        String inputContent = messageRequest.getContent();
        MessageGenerationResult messageGenerationResult = messageGenerator.generateMessage(inputContent);

        message.put(PayloadFieldTypes.TEXT, messageGenerationResult.getMessage());
        response.put(PayloadFieldTypes.MESSAGE, message);

        // 메세지 생성에 성공했을 경우에만 데이터베이스에 기록을 남긴다.
        if (messageGenerationResult.isValid()) {
            AlpacapacaRecord record = new AlpacapacaRecord();
            record.setUserKey(messageRequest.getUserKey());
            record.setInput(inputContent);
            record.setOutput(messageGenerationResult.getMessage());
            record.setRequestTime(serverTime);
            recordRepository.save(record);
        }

        return response;
    }
}
