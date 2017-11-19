package com.ybigta.alpacapaca.autoreply.service.validator;

import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.Matchers.is;
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
        boolean trueExpectedValidationResult = contentValidator.validate(koreanStringLength3);

        // then
        assertThat(trueExpectedValidationResult, is(true));
    }

    @Test
    public void testReturnFalseIfInputIsKoreanLengthIsNot3() {
        // given
        String koreanStringLength4 = "한글임다";
        String koreanStringLength2 = "한글";

        // when
        boolean falseExpectedValidationResultDueToLengthIs4 = contentValidator.validate(koreanStringLength4);
        boolean falseExpectedValidationResultDueToLengthIs2 = contentValidator.validate(koreanStringLength2);

        // then
        assertThat(falseExpectedValidationResultDueToLengthIs4, is(false));
        assertThat(falseExpectedValidationResultDueToLengthIs2, is(false));
    }

    @Test
    public void testReturnFalseIfInputIsEnglishLengthIs3() {
        // given
        String englishStringLength3 = "abc";

        // when
        boolean falseExpectedValidationResultDueToEnglish = contentValidator.validate(englishStringLength3);

        // then
        assertThat(falseExpectedValidationResultDueToEnglish, is(false));
    }

    @Test
    public void testReturnFalseIfInputIsNumberLengthIs3() {
        // given
        String numericStringLength3 = "123";

        // when
        boolean falseExpectedValidationResultDueToNumeric = contentValidator.validate(numericStringLength3);

        // then
        assertThat(falseExpectedValidationResultDueToNumeric, is(false));
    }
}
