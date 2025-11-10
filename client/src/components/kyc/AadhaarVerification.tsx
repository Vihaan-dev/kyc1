import React from "react";
import styled from "styled-components";
import { Button } from "../ui/button";
import "@/components/translations/Translations";
import { useTranslation } from "react-i18next";

const AadhaarContainer = styled.div`
  text-align: center;
  margin-top: 30px;
  margin-bottom: 30px;
`;

const AadhaarDetails = styled.div`
  background-color: #f4f4f4;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Field = styled.div`
  margin-bottom: 10px;
`;

const Label = styled.span`
  font-weight: bold;
`;

const Value = styled.span`
  margin-left: 10px;
`;

export default function AadhaarVerification({
  onNextStep,
}: {
  onNextStep: () => void;
}) {
  const { t } = useTranslation();
  const speakMessage = (message) => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      const speech = new SpeechSynthesisUtterance();
      speech.text = message;
      speech.volume = 1;
      speech.rate = 1;
      speech.pitch = 1;
      window.speechSynthesis.speak(speech);
    }
  };

  const onSubmit = () => {
    onNextStep();
    speakMessage(
      "Position your face in the center of the frame and click the start button to begin the test.",
    );
  };

  const data = JSON.parse(localStorage.getItem("aadhar-data") || "{}");

  // Simple display - just show what was extracted
  const displayData = {
    aadhaar_number: data.aadhaar_number || data.partial_data?.aadhaar_number || 'Not extracted',
    name: data.name || data.partial_data?.name || 'Not extracted',
    dob: data.dob || data.partial_data?.dob || 'Not extracted',
    gender: data.gender || data.partial_data?.gender || 'Not extracted',
    address: data.address || data.partial_data?.address || 'Not extracted',
    manualReview: data.manual_review || false
  };

  const hasData = displayData.name !== 'Not extracted' || displayData.aadhaar_number !== 'Not extracted';

  return (
    <AadhaarContainer>
      <h2 className="text-lg font-semibold mb-4">
        {t("Here are the details we fetched from your Aadhaar card:")}
      </h2>
      
      {displayData.manualReview && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <p className="text-yellow-800 text-sm">
            ⚠️ Some data may require manual verification. Please review carefully.
          </p>
        </div>
      )}
      
      <AadhaarDetails>
        <Field>
          <Label>Aadhaar Number:</Label>
          <Value>{displayData.aadhaar_number}</Value>
        </Field>
        <Field>
          <Label>Name:</Label>
          <Value>{displayData.name}</Value>
        </Field>
        <Field>
          <Label>Date of Birth:</Label>
          <Value>{displayData.dob}</Value>
        </Field>
        <Field>
          <Label>Gender:</Label>
          <Value>{displayData.gender}</Value>
        </Field>
        <Field>
          <Label>Address:</Label>
          <Value>{displayData.address}</Value>
        </Field>
      </AadhaarDetails>
      
      {!hasData && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4">
          <p className="text-red-800 text-sm">
            ⚠️ No data could be extracted. Please proceed for manual verification.
          </p>
        </div>
      )}
      
      <Button className="my-10 bg-blue-600" onClick={onSubmit}>
        {t("Verify & Continue")}
      </Button>
    </AadhaarContainer>
  );
}
