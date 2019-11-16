import request from "supertest";
import app from "../src/app";
import { parseMessage } from "../src/public/js/parse";

describe("GET /api", () => {
  it("should return 200 OK", () => {
    return request(app)
      .get("/api")
      .expect(200);
  });
});

describe("Parse FAIRSEQ output", () => {
  it("should return a list of sentences", () => {
    const message = `
        S-272120        german stocks closed down wednesday at the frankfurt stock exchange .
        T-272120        german stocks close lower
        H-272120        -0.9572659730911255     german stocks close down at frankfurt bourse
        P-272120        0.0000 -0.3565 -0.7092 -0.7942 -0.9684 -0.2987 -1.5121 -3.9764 0.0000
        I-272120        2
        E-272120_0
        E-272120_1      <unk> <unk> <unk> <unk> <unk> <unk>
        E-272120_2      german stocks close down at frankfurt
        E-272120_3      german stocks close down at frankfurt
        E-272120_4      german stocks close down at frankfurt <unk>
        E-272120_5      german stocks close down at frankfurt bourse
        E-272120_6      german stocks close down at frankfurt bourse
        E-272120_7      german stocks close down at frankfurt bourse
        E-272120_8      german stocks close down at frankfurt bourse
        `;
    console.log(parseMessage(message));
  });
});
