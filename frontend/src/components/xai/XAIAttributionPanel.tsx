"use client";

import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, ShieldCheck, UserCheck } from "lucide-react";

/**
 * XAI Attribution Panel (Iteration 5)
 * ====================================
 * Displays the Eigen-CAM heatmap overlay + conformal prediction set
 * + route-to-human badge from the Iteration 4 ModalityResult fields.
 *
 * Research grounding:
 * - Eigen-CAM (Muhammad & Yeasin, IJCNN 2020): gradient-free SVD-based
 *   attribution. Displayed as a heatmap overlay on the analyzed image.
 * - Conformal RAPS (Romano et al., ICLR 2021): distribution-free
 *   prediction sets. {real} or {fake} = confident; {real, fake} =
 *   ambiguous → route to human.
 *
 * Props match the ModalityResult.xai_attribution + conformal_prediction_set
 * + route_to_human fields from the backend schema.
 */

interface XAIAttributionData {
  method: string;
  heatmap: number[][];
  heatmap_shape: [number, number];
  explanation: string;
  verdict: string;
  score: number;
}

interface XAIAttributionPanelProps {
  xai_attribution: XAIAttributionData | null;
  conformal_prediction_set: number[] | null;
  route_to_human: boolean;
  imageUrl?: string;
  className?: string;
}

export function XAIAttributionPanel({
  xai_attribution,
  conformal_prediction_set,
  route_to_human,
  imageUrl,
  className,
}: XAIAttributionPanelProps) {
  // Build a CSS-grid heatmap visualization from the 28x28 data
  const heatmapStyle = useMemo(() => {
    if (!xai_attribution?.heatmap) return null;
    const { heatmap, heatmap_shape } = xai_attribution;
    const [rows, cols] = heatmap_shape;
    return {
      display: "grid",
      gridTemplateColumns: `repeat(${cols}, 1fr)`,
      gridTemplateRows: `repeat(${rows}, 1fr)`,
      gap: "0",
      width: "100%",
      aspectRatio: `${cols} / ${rows}`,
    } as React.CSSProperties;
  }, [xai_attribution]);

  const conformalLabel = useMemo(() => {
    if (!conformal_prediction_set) return null;
    if (conformal_prediction_set.length === 1) {
      return conformal_prediction_set[0] === 0
        ? { text: "Confident: Real", color: "green" }
        : { text: "Confident: Fake", color: "red" };
    }
    return { text: "Ambiguous: Route to Human", color: "yellow" };
  }, [conformal_prediction_set]);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ShieldCheck className="h-5 w-5 text-blue-500" />
          Explainability & Conformal Prediction
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Route-to-human badge */}
        {route_to_human && (
          <div className="flex items-center gap-2 rounded-lg bg-yellow-50 border border-yellow-200 p-3">
            <UserCheck className="h-5 w-5 text-yellow-600" />
            <div>
              <p className="text-sm font-medium text-yellow-900">
                Routed to Human Review
              </p>
              <p className="text-xs text-yellow-700">
                Conformal prediction set is ambiguous or adversarial gate
                triggered. This input requires manual verification.
              </p>
            </div>
          </div>
        )}

        {/* Conformal prediction set badge */}
        {conformalLabel && (
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">
              Conformal Prediction:
            </span>
            <Badge
              variant={
                conformalLabel.color === "green"
                  ? "default"
                  : conformalLabel.color === "red"
                  ? "destructive"
                  : "secondary"
              }
              className={
                conformalLabel.color === "green"
                  ? "bg-green-600"
                  : conformalLabel.color === "yellow"
                  ? "bg-yellow-500"
                  : ""
              }
            >
              {conformalLabel.text}
            </Badge>
            <span className="text-xs text-gray-500">
              (90% coverage, α=0.10)
            </span>
          </div>
        )}

        {/* Eigen-CAM heatmap */}
        {xai_attribution && heatmapStyle && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700">
                Attribution Heatmap ({xai_attribution.method}):
              </span>
              <Badge variant="outline" className="text-xs">
                {xai_attribution.verdict === "fake"
                  ? `Fake (${xai_attribution.score.toFixed(3)})`
                  : `Real (${xai_attribution.score.toFixed(3)})`}
              </Badge>
            </div>

            {/* Heatmap grid */}
            <div
              className="relative rounded-lg overflow-hidden border border-gray-200"
              style={heatmapStyle}
            >
              {xai_attribution.heatmap.map((row, i) =>
                row.map((value, j) => {
                  // Map [0,1] to a red-yellow-green colormap
                  // High value = red (high influence), low = green
                  const hue = (1 - value) * 120; // 0=red, 120=green
                  return (
                    <div
                      key={`${i}-${j}`}
                      style={{
                        backgroundColor: `hsl(${hue}, 80%, 50%)`,
                        opacity: 0.3 + value * 0.7,
                      }}
                    />
                  );
                })
              )}
            </div>

            {/* Color legend */}
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span>Low influence</span>
              <div
                className="h-3 flex-1 rounded"
                style={{
                  background:
                    "linear-gradient(to right, hsl(120,80%,50%), hsl(60,80%,50%), hsl(0,80%,50%))",
                }}
              />
              <span>High influence</span>
            </div>

            {/* Human-readable explanation */}
            <p className="text-xs text-gray-600 leading-relaxed bg-gray-50 rounded p-2">
              {xai_attribution.explanation}
            </p>
          </div>
        )}

        {/* No data fallback */}
        {!xai_attribution && !conformalLabel && !route_to_human && (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <AlertTriangle className="h-4 w-4" />
            <span>
              XAI attribution not available for this analysis. Enable
              ENABLE_XAI_ATTRIBUTION_OUTPUT=true to generate.
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
