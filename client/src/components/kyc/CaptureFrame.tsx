import React, { useState, useRef, useEffect } from "react";
import WebcamFeed from "@/components/kyc/WebcamFeed";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import "@/components/translations/Translations";
import { useTranslation } from "react-i18next";
import "@mediapipe/face_mesh";
import "@mediapipe/camera_utils";

const CaptureFrame = ({ onNextStep }) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const blinkCounter = useRef(0);
  const turnLeftCounter = useRef(0);
  const turnRightCounter = useRef(0);
  const selfieCaptured = useRef(false);
  const stageAnnounced = useRef({
    blink: false,
    turnLeft: false,
    turnRight: false,
    selfie: false,
  });
  // Step tracking: instructions -> blink -> turn -> selfie -> done
  const [livenessStage, setLivenessStage] = useState<"instructions" | "blink" | "turnLeft" | "turnRight" | "selfie" | "done">("instructions");
  const [capturedImages, setCapturedImages] = useState({ photo: null });
  const [title, setTitle] = useState("Video Verification");
  const [subtitle, setSubtitle] = useState("Click start to begin your liveness test");

  const [showLoader, setShowLoader] = useState(false);

  const speakMessage = (message: string) => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      const speech = new SpeechSynthesisUtterance();
      speech.text = message;
      speech.volume = 1;
      speech.rate = 1;
      speech.pitch = 1;
      window.speechSynthesis.speak(speech);
    }
  };

  // Core image capture logic
  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const context = canvas.getContext("2d");
      const rectX = video.videoWidth * 0.1;
      const rectY = video.videoHeight * 0.1;
      const rectWidth = video.videoWidth * 0.8;
      const rectHeight = video.videoHeight * 0.8;
      canvas.width = rectWidth;
      canvas.height = rectHeight;
      context.drawImage(video, rectX, rectY, rectWidth, rectHeight, 0, 0, rectWidth, rectHeight);
      const imageDataUrl = canvas.toDataURL("image/jpg");
      setCapturedImages({ photo: imageDataUrl });
      return imageDataUrl;
    }
    return null;
  };

  // upload captured selfie to the backend
  const uploadSelfie = async (dataUrl: string) => {
    try {
      setShowLoader(true);
      // Convert dataURL to blob
      const res = await fetch(dataUrl);
      const blob = await res.blob();
      const file = new File([blob], "selfie.jpg", { type: "image/jpg" });

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:5002/livephoto-upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("Backend response:", data);
      setShowLoader(false);
      return data;
    } catch (err) {
      console.error("Error uploading selfie:", err);
      setShowLoader(false);
    }
  };

  useEffect(() => {
    canvasRef.current = document.createElement("canvas");
  }, []);

  // Initialize FaceMesh for liveness
  useEffect(() => {

    if (livenessStage === "instructions") return;
    if (typeof window === "undefined" || !videoRef.current) return;

    let faceMesh: any;
    let camera: any;

    const initFaceMesh = async () => {
      const { FaceMesh } = await import("@mediapipe/face_mesh");
      const { Camera } = await import("@mediapipe/camera_utils");

      const video = videoRef.current;
      if (!video) {
        console.warn("Video element not found yet");
        return;
      }

      faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults(onResults);

      const startProcessing = () => {
        console.log("Video is ready, starting FaceMesh...");
        camera = new Camera(video, {
          onFrame: async () => {
            if (video.videoWidth === 0 || video.videoHeight === 0) return; // skip if not ready
            await faceMesh.send({ image: video });
          },
          width: 640,
          height: 480,
        });
        camera.start();
      };

      if (video.readyState >= 2) {
        startProcessing();
      } else {
        video.addEventListener("loadeddata", startProcessing, { once: true });
      }
    };

    const onResults = (results: any) => {
      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
      const landmarks = results.multiFaceLandmarks[0];

      //Eye blink detection (compare top and bottom eyelid distance)
      const leftEyeTop = landmarks[159].y;
      const leftEyeBottom = landmarks[145].y;
      const rightEyeTop = landmarks[386].y;
      const rightEyeBottom = landmarks[374].y;
      const leftEyeDistance = Math.abs(leftEyeTop - leftEyeBottom);
      const rightEyeDistance = Math.abs(rightEyeTop - rightEyeBottom);
      const noseX = landmarks[1].x;

      switch (livenessStage) {

        case "blink":
          if (leftEyeDistance < 0.01 && rightEyeDistance < 0.01) {
            blinkCounter.current += 1;
          } else {
            blinkCounter.current = 0;
          }
          if (blinkCounter.current > 3 && !stageAnnounced.current.blink) { // stable blink for 3 frames
            speakMessage("Good job! Now turn your head to the left.");
            stageAnnounced.current.blink = true;
            setLivenessStage("turnLeft");
            blinkCounter.current = 0;
          }
          break;

        case "turnLeft":
          if (noseX > 0.6) turnLeftCounter.current += 1;
          else turnLeftCounter.current = 0;

          if (turnLeftCounter.current > 5 && !stageAnnounced.current.turnLeft) { // stable turn for 5 frames
            speakMessage("Great! Now turn your head to the right.");
            stageAnnounced.current.turnLeft = true;
            setLivenessStage("turnRight");
            turnLeftCounter.current = 0;
          }
          break;

        case "turnRight":
          if (noseX < 0.4) turnRightCounter.current += 1;
          else turnRightCounter.current = 0;

          if (turnRightCounter.current > 5 && !stageAnnounced.current.turnRight) { // stable turn for 5 frames
            speakMessage("Perfect! Hold still for a moment and smile.");
            stageAnnounced.current.turnRight = true;
            setLivenessStage("selfie");
            turnRightCounter.current = 0;
          }
          break;

        case "selfie":
          if (!selfieCaptured.current && !stageAnnounced.current.selfie) {
            stageAnnounced.current.selfie = true;
            selfieCaptured.current = true;

            (async () => {
              // Give user 4 seconds to pose
              await new Promise((resolve) => setTimeout(resolve, 4000));

              const dataUrl = captureImage(); // capture image returns dataUrl
              if (dataUrl) {
                try {
                  await uploadSelfie(dataUrl); // upload to backend
                } catch (err) {
                  console.error("Error uploading selfie:", err);
                  speakMessage("Failed to upload selfie. Please try again.");
                  return; // stop here if upload fails
                }
              }

              speakMessage("Liveness test completed successfully!");
              setLivenessStage("done");
            })();
          }
          break;

        case "done":
          // nothing to do
          break;
      }
    };

    initFaceMesh();

    return () => {
      if (camera) camera.stop();
    };
  }, [livenessStage]);

  // UI Rendering
  return (
    <div className="text-center w-[600px] mx-auto my-10 p-5">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="text-md">{subtitle}</p>

      {/* Webcam or Captured Selfie */}
      {capturedImages.photo ? (
        <Image
          src={capturedImages.photo}
          alt="Captured Selfie"
          className="mx-auto border border-gray-300 rounded-sm my-5"
          width={560}
          height={430}
        />
      ) : (
        <WebcamFeed videoRef={videoRef} frameType="selfie"/>
      )}

      {/* Buttons */}
      <div className="button-container flex justify-center mt-4">
        {!capturedImages.photo ? (
          <Button
            onClick={() => {
              setLivenessStage("blink");
              speakMessage("Please blink your eyes.");
              setSubtitle("Blink your eyes when ready");
            }}
            variant="outline"
            className="bg-blue-600 text-white"
          >
            Start Liveness Test
          </Button>
        ) : (
          <Button
            onClick={() => onNextStep()}
            variant="outline"
            className="bg-green-500 text-white"
          >
            Continue
          </Button>
        )}
      </div>
    </div>
  );
};

export default CaptureFrame;
