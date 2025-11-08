import { Request, Response } from 'express';
import { Signup, signupSchema, verifyPhoneSchema, VerifyPhone, verifyEmailSchema, VerifyEmail, signInSchema, SignIn } from './auth.schema';
import {
  handleForgotPassword,
  handleSendOTP,
  handleSignIn,
  handleSignUp,
  handleVerifyEmail,
  handleVerifyPhone,
} from './auth.service';

export const signuUp = async (req: Request, res: Response) => {
  try {
    const validatedData = signupSchema.parse(req.body);
    const signup = new Signup(validatedData);
    await handleSignUp(signup);
    res.status(201).json({ message: 'OTP sent successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const verifyPhone = async (req: Request, res: Response) => {
  try {
    const validatedData = verifyPhoneSchema.parse(req.body);
    const verifyPhone = new VerifyPhone(validatedData);
    await handleVerifyPhone(verifyPhone);
    res.status(200).json({ message: 'Phone number verified successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const verifyEmail = async (req: Request, res: Response) => {
  try {
    const validatedData = verifyEmailSchema.parse(req.body);
    const verifyEmail = new VerifyEmail(validatedData);
    await handleVerifyEmail(verifyEmail);
    res.status(200).json({ message: 'Email verified successfully' });
  } catch (error) {
    console.error('ERROR', error);
    res.status(400).json({ error: error.message });
  }
};

export const forgotPassword = async (req: Request, res: Response) => {
  try {
    const { password } = req.body;
    const { phone } = req.params;
    await handleForgotPassword({ phone, password });
    res.status(200).json({ message: 'Password updated successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const sendOTP = async (req: Request, res: Response) => {
  try {
    const { device } = req.body;
    await handleSendOTP(device);
    res.status(200).json({ message: `OTP sent to ${device}` });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const signIn = async (req: Request, res: Response) => {
  try {
    const validatedData = signInSchema.parse(req.body);
    const signIn = new SignIn(validatedData);
    const user = await handleSignIn(signIn.email, signIn.password);
    res.status(200).json({ message: 'Signed in successfully', data: { user } });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};
