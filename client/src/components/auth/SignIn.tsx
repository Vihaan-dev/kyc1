import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import Link from "next/link";
import { useRef, useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";

const formSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8, {
    message: "Please enter a valid password",
  }),
});

// Google reCAPTCHA test key for localhost - always passes
// Replace with your production key: 6Ld1hQYsAAAAALILNYdNp8_FjSDYCIqB-w3L8Aop
const RECAPTCHA_SITE_KEY = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI";

export default function SignInForm({ onSuccess }: { onSuccess: () => void }) {
  const recaptchaRef = useRef<ReCAPTCHA>(null);
  const [recaptchaToken, setRecaptchaToken] = useState<string | null>(null);
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      // Validate reCAPTCHA
      if (!recaptchaToken) {
        alert("Please complete the reCAPTCHA verification");
        return;
      }

      const response = await fetch("http://localhost:5001/api/auth/signin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...values,
          recaptchaToken,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit form");
      }

      const {
        data: { user },
      } = await response.json();

      localStorage.setItem("email", user.email);
      localStorage.setItem("phone", user.phone);

      // Handle success response
      console.log("Form submitted successfully");
      form.reset();
      recaptchaRef.current?.reset();
      setRecaptchaToken(null);
      onSuccess(); // Call the onSuccess function here
    } catch (error) {
      console.error("Error submitting form:", error);
      // Reset reCAPTCHA on error
      recaptchaRef.current?.reset();
      setRecaptchaToken(null);
    }
  };

  const onRecaptchaChange = (token: string | null) => {
    setRecaptchaToken(token);
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="flex flex-col gap-5 w-1/3 mx-auto justify-center min-h-screen"
      >
        <div className="text-center">
          <h1 className="font-bold text-xl">Welcome back</h1>
          <p className="text-sm">
            Fill your credentials to login to your account
          </p>
        </div>
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Input placeholder="Enter your email address" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input
                  type="password"
                  placeholder="Enter your password"
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        {/* reCAPTCHA */}
        <div className="flex justify-center">
          <ReCAPTCHA
            ref={recaptchaRef}
            sitekey={RECAPTCHA_SITE_KEY}
            onChange={onRecaptchaChange}
          />
        </div>
        
        <Button type="submit" disabled={!recaptchaToken}>
          Next
        </Button>
        <Link href="/" className="text-xs self-center underline">
          New here? Sign Up
        </Link>
      </form>
    </Form>
  );
}
