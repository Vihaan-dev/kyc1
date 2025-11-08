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
import { useRef, useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";

const formSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8, {
    message: "Password must be at least 8 characters.",
  }),
  phone: z
    .string()
    .length(13, { message: "Phone number must be 13 characters." }),
});

// Google reCAPTCHA test key for localhost - always passes
// Replace with your production key: 6Ld1hQYsAAAAALILNYdNp8_FjSDYCIqB-w3L8Aop
const RECAPTCHA_SITE_KEY = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI";

export default function SignUpForm() {
  const recaptchaRef = useRef<ReCAPTCHA>(null);
  const [recaptchaToken, setRecaptchaToken] = useState<string | null>(null);
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
      phone: "",
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      // Validate reCAPTCHA
      if (!recaptchaToken) {
        alert("Please complete the reCAPTCHA verification");
        return;
      }

      // const response = await fetch("/kycDetails", {
      //     method: "POST",
      //     headers: {
      //         "Content-Type": "application/json",
      //     },
      //     body: JSON.stringify({
      //       ...values,
      //       recaptchaToken,
      //     }),
      // });

      // if (!response.ok) {
      //     throw new Error("Failed to submit form");
      // }
      // Handle success response
      console.log("Form submitted successfully with reCAPTCHA token");
      window.location.href = "/otp-verify";
      form.reset();
      recaptchaRef.current?.reset();
      setRecaptchaToken(null);
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
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Email</FormLabel>
              <FormControl>
                <Input placeholder="Email" {...field} />
              </FormControl>
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
                <Input type="password" placeholder="Password" {...field} />
              </FormControl>
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="phone"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Phone</FormLabel>
              <FormControl>
                <Input placeholder="Phone" {...field} />
              </FormControl>
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
        
        <FormMessage />
        <Button type="submit" disabled={!recaptchaToken}>
          Next
        </Button>
      </form>
    </Form>
  );
}
