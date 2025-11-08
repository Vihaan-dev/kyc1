import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "../ui/button";
import { useState } from "react";
import "@/components/translations/Translations";
import { useTranslation } from "react-i18next";

export default function UploadPhotoFile() {
  const { t } = useTranslation();
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [attemptCount, setAttemptCount] = useState(0);
  const [extractionStatus, setExtractionStatus] = useState<'idle' | 'success' | 'failed'>('idle');
  const [message, setMessage] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleReupload = () => {
    setIsUploaded(false);
    setSelectedFile(null);
    setExtractionStatus('idle');
    setMessage("");
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      console.log("No file selected");
      return;
    }
    
    const currentAttempt = attemptCount + 1;
    setAttemptCount(currentAttempt);
    
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await fetch("http://localhost:5002/livephoto-upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log("data", data);
      
      // Check if face matching was successful
      if (data.face_matching === true) {
        setIsUploaded(true);
        setExtractionStatus('success');
        setMessage("✓ Photo uploaded and face matched successfully!");
      } else if (data.face_matching === false) {
        // Face matching failed
        setIsUploaded(true);
        setExtractionStatus('failed');
        
        if (currentAttempt < 2) {
          setMessage(`⚠ Face matching failed (Attempt ${currentAttempt}/2). Please ensure your face is clearly visible and matches Aadhaar photo.`);
        } else {
          setMessage("⚠ Face matching failed after 2 attempts. Photo saved for manual review. You may proceed.");
        }
      } else {
        // No face detected or other error
        setIsUploaded(true);
        setExtractionStatus('failed');
        
        if (currentAttempt < 2) {
          setMessage(`⚠ Could not process photo (Attempt ${currentAttempt}/2). Please upload a clear passport-size photo.`);
        } else {
          setMessage("⚠ Photo processing failed after 2 attempts. Photo saved for manual review. You may proceed.");
        }
      }
    } catch (error) {
      console.log("error", error);
      setIsUploaded(true);
      setExtractionStatus('failed');
      
      if (attemptCount + 1 < 2) {
        setMessage(`✗ Upload failed (Attempt ${attemptCount + 1}/2). Please try again.`);
      } else {
        setMessage("✗ Upload failed after 2 attempts. Photo saved for manual review. You may proceed.");
      }
    }
  };
  
  return (
    <form onSubmit={onSubmit} encType="multipart/form-data">
      <Label htmlFor="photo">{t("Upload Passport Size Photo")}</Label>
      <Input type="file" name="photo" id="photo" onChange={handleFileChange} />
      
      {isUploaded ? (
        <div className="space-y-2 mt-2">
          <p className={
            extractionStatus === 'success' ? 'text-green-600' : 
            attemptCount >= 2 ? 'text-orange-600' : 'text-red-600'
          }>
            {message}
          </p>
          
          {extractionStatus === 'failed' && attemptCount < 2 && (
            <Button 
              type="button" 
              onClick={handleReupload} 
              className="w-full"
              variant="outline"
            >
              {t("Re-upload Photo")}
            </Button>
          )}
          
          {(extractionStatus === 'success' || attemptCount >= 2) && (
            <p className="text-sm text-gray-600">
              {extractionStatus === 'success' 
                ? t("File uploaded successfully") 
                : t("Proceed to next step - file will be reviewed manually")}
            </p>
          )}
        </div>
      ) : (
        <Button type="submit" className="w-full mt-2">
          {t("Submit")}
        </Button>
      )}
    </form>
  );
}
