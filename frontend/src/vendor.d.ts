// Type stubs for optional third-party packages not yet on DefinitelyTyped.
// Remove when official @types packages become available.

declare module "@sparkjsdev/spark" {
  export class Viewer {
    constructor(opts: {
      container: HTMLElement;
      maxSplats?: number;
      camera?: { position?: number[]; target?: number[]; up?: number[] };
    });
    load(url: string): Promise<void>;
    dispose(): void;
    stop(): void;
  }
}
