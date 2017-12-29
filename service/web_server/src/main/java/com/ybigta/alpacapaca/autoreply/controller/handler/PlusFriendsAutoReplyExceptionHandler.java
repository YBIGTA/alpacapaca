package com.ybigta.alpacapaca.autoreply.controller.handler;

import com.ybigta.alpacapaca.autoreply.PayloadFieldTypes;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestControllerAdvice
public class PlusFriendsAutoReplyExceptionHandler {
    private static final String SERVER_ERROR_MESSAGE = "죄송합니다. 서버 문제로 메세지 생성에 실패하였습니다.";

    @ExceptionHandler({Exception.class})
    @ResponseStatus(HttpStatus.OK)
    public Map<String, Object> handleDefaultError(final Exception exception) {
        log.error(exception.getMessage());

        Map<String, Object> response = new HashMap<>();
        Map<String, Object> message = new HashMap<>();

        message.put(PayloadFieldTypes.TEXT, SERVER_ERROR_MESSAGE);
        response.put(PayloadFieldTypes.MESSAGE, message);

        return response;
    }
}
