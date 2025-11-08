import axios from 'axios';

// Google reCAPTCHA test secret for localhost - always passes
// Replace with your production secret: 6Ld1hQYsAAAAAGXyvazXs9M2BBU8XW1hUn5ayOyn
const RECAPTCHA_SECRET_KEY = '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe';

export interface RecaptchaVerificationResponse {
  success: boolean;
  challenge_ts?: string;
  hostname?: string;
  'error-codes'?: string[];
}

export const verifyRecaptcha = async (token: string): Promise<boolean> => {
  try {
    const response = await axios.post<RecaptchaVerificationResponse>(
      'https://www.google.com/recaptcha/api/siteverify',
      null,
      {
        params: {
          secret: RECAPTCHA_SECRET_KEY,
          response: token,
        },
      }
    );

    return response.data.success;
  } catch (error) {
    console.error('reCAPTCHA verification error:', error);
    return false;
  }
};
