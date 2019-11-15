"use strict";

import { Response, Request } from "express";
export function getDashboard(req: Request, res: Response) {
  res.render("dashboard/index", {
    title: "Levenshtein Demo"
  });
}
