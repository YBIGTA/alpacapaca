package com.ybigta.alpacapaca.autoreply.service.validator;

import com.ybigta.alpacapaca.autoreply.service.ErrorMessages;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThat;

public class ValidationImplTests {

    private ContentValidatorImpl contentValidator;

    @Before
    public void setUp() {
        contentValidator = new ContentValidatorImpl();
    }

    @Test
    public void testReturnTrueIfInputIsKoreanLengthIs3() {
        // given
        String koreanStringLength3 = "한글임";

        // when
        ValidationResult trueExpectedValidationResult = contentValidator.validate(koreanStringLength3);

        // then
        assertThat(trueExpectedValidationResult.isValid(), is(true));
        assertNull(trueExpectedValidationResult.getMessage());
    }

    @Test
    public void testReturnFalseIfInputIsNullString() {
        // given
        String nullString = null;

        // when
        ValidationResult falseExpectedValidationResultDueToNull = contentValidator.validate(nullString);

        // then
        assertThat(falseExpectedValidationResultDueToNull.isValid(), is(false));
        assertThat(falseExpectedValidationResultDueToNull.getMessage(), is(ErrorMessages.INPUT_MESSAGE_IS_NULL));
    }

    @Test
    public void testReturnFalseIfInputIsKoreanLengthIsNot3() {
        // given
        String koreanStringLength4 = "한글임다";
        String koreanStringLength2 = "한글";

        // when
        ValidationResult falseExpectedValidationResultDueToLengthIs4 = contentValidator.validate(koreanStringLength4);
        ValidationResult falseExpectedValidationResultDueToLengthIs2 = contentValidator.validate(koreanStringLength2);

        // then
        assertThat(falseExpectedValidationResultDueToLengthIs2.isValid(), is(false));
        assertThat(falseExpectedValidationResultDueToLengthIs2.getMessage(), is(ErrorMessages.LENGTH_IS_NOT_VALID_MESSAGE));
        assertThat(falseExpectedValidationResultDueToLengthIs4.isValid(), is(false));
        assertThat(falseExpectedValidationResultDueToLengthIs4.getMessage(), is(ErrorMessages.LENGTH_IS_NOT_VALID_MESSAGE));
    }

    @Test
    public void testReturnFalseIfInputIsEnglishLengthIs3() {
        // given
        String englishStringLength3 = "abc";

        // when
        ValidationResult falseExpectedValidationResultDueToEnglish = contentValidator.validate(englishStringLength3);

        // then
        assertThat(falseExpectedValidationResultDueToEnglish.isValid(), is(false));
        assertThat(falseExpectedValidationResultDueToEnglish.getMessage(), is(ErrorMessages.INPUT_CONTENT_IS_NOT_KOREAN_MESSAGE));
    }

    @Test
    public void testReturnFalseIfInputIsNumberLengthIs3() {
        // given
        String numericStringLength3 = "123";

        // when
        ValidationResult falseExpectedValidationResultDueToNumeric = contentValidator.validate(numericStringLength3);

        // then
        assertThat(falseExpectedValidationResultDueToNumeric.isValid(), is(false));
        assertThat(falseExpectedValidationResultDueToNumeric.getMessage(), is(ErrorMessages.INPUT_CONTENT_IS_NOT_KOREAN_MESSAGE));
    }
}
