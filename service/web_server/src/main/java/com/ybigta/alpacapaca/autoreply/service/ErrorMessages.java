package com.ybigta.alpacapaca.autoreply.service;

public class ErrorMessages {
    public static final String INPUT_MESSAGE_IS_NULL = "메세지를 입력하지 않으셨습니다.";
    public static final String LENGTH_IS_NOT_VALID_MESSAGE = "입력하신 메세지가 3글자가 아닙니다.";
    public static final String INPUT_CONTENT_IS_NOT_KOREAN_MESSAGE = "입력하신 메세지가 한글 단어가 아닙니다. (자음어는 지원하지 않습니다)";

    public static final String SERVER_ERROR_MESSAGE = "죄송합니다. 서버 문제로 메세지 생성에 실패하였습니다.";
    public static final String MESSAGE_GENERATING_FAILURE_MESSAGE = "죄송합니다. 입력해주신 단어로 응답 메세지 생성에 실패하였습니다.";

    private ErrorMessages() {
    }
}
