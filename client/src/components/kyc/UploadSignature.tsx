import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "../ui/button";
import { useState } from "react";
import "@/components/translations/Translations";
import { useTranslation } from "react-i18next";

export default function UploadSignatureFile() {
  const { t } = useTranslation();
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSelectedFile(e.target.files[0]);
      setIsUploaded(false);
      setMessage("");
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      console.log("No file selected");
      return;
    }
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await fetch("http://localhost:5002/signature-upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log("data", data);
      
      if (res.ok) {
        setIsUploaded(true);
        setMessage("✓ Signature uploaded successfully");
      } else {
        setMessage("✗ Upload failed. Please try again.");
      }
    } catch (error) {
      console.log("error", error);
      setMessage("✗ Upload failed. Please try again.");
    }
  };
  
  return (
    <form onSubmit={onSubmit} encType="multipart/form-data">
      <Label htmlFor="signature">{t("Upload Signature")}</Label>
      <Input type="file" name="signature" id="signature" onChange={handleFileChange} />
      
      {isUploaded ? (
        <div className="space-y-2 mt-2">
          <p className="text-green-600">{message}</p>
          <p className="text-sm text-gray-600">{t("File uploaded successfully")}</p>
        </div>
      ) : message ? (
        <div className="space-y-2 mt-2">
          <p className="text-red-600">{message}</p>
          <Button type="submit" className="w-full">
            {t("Retry")}
          </Button>
        </div>
      ) : (
        <Button type="submit" className="w-full mt-2">
          {t("Submit")}
        </Button>
      )}
    </form>
  );
}
