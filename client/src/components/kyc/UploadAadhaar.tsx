import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "../ui/button";
import { useState } from "react";
import { xmlToJson } from "@/utils";
import "@/components/translations/Translations";
import { useTranslation } from "react-i18next";

export default function UploadAadhaarFile() {
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
      const res = await fetch("http://localhost:5002/aadhar-upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log("Aadhaar response:", data);
      
      // Check if extraction was successful
      if (data.success && (data.name || data.aadhaar_number)) {
        // Store extracted data
        const extractedData = {
          aadhaar_number: data.aadhaar_number,
          name: data.name,
          dob: data.dob,
          gender: data.gender,
          address: data.address,
          photo: data.photo
        };
        
        localStorage.setItem("aadhar-data", JSON.stringify(extractedData));
        console.log("Stored Aadhaar data:", extractedData);
        setIsUploaded(true);
        setExtractionStatus('success');
        
        const displayInfo = data.name || data.aadhaar_number || 'Data extracted';
        setMessage(`✓ Aadhaar extracted successfully: ${displayInfo}`);
      } else {
        // Extraction failed
        setIsUploaded(true);
        setExtractionStatus('failed');
        
        if (currentAttempt < 2) {
          setMessage(`⚠ Data extraction failed (Attempt ${currentAttempt}/2). Please re-upload a clearer Aadhaar image.`);
        } else {
          // After 2 attempts, save partial data for manual review
          localStorage.setItem("aadhar-data", JSON.stringify({ 
            manual_review: true,
            partial_data: {
              aadhaar_number: data.aadhaar_number || '',
              name: data.name || '',
              dob: data.dob || '',
              gender: data.gender || '',
              address: data.address || ''
            }
          }));
          setMessage(`⚠ Extraction incomplete after 2 attempts. Image saved for manual review. You may proceed.`);
        }
      }
    } catch (error) {
      console.log("error", error);
      setIsUploaded(true);
      setExtractionStatus('failed');
      
      if (attemptCount + 1 < 2) {
        setMessage(`✗ Upload failed (Attempt ${attemptCount + 1}/2). Please try again.`);
      } else {
        localStorage.setItem("aadhar-data", JSON.stringify({ manual_review: true }));
        setMessage("✗ Upload failed after 2 attempts. Image saved for manual review. You may proceed.");
      }
    }
  };  return (
    <form onSubmit={onSubmit} encType="multipart/form-data">
      <Label htmlFor="aadhar">{t("Upload Aadhar")}</Label>
      <Input type="file" name="aadhar" id="aadhar" onChange={handleFileChange} />
      
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
              {t("Re-upload Aadhar Card")}
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
