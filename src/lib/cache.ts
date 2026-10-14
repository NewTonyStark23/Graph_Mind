// IndexedDB wrapper for caching extracted model data
// Provides persistent storage for parsed models, statistics, and brain maps

const DB_NAME = 'GraphifyCache';
const DB_VERSION = 1;

export interface CachedModel {
  id: string;
  modelId: string;
  fileName: string;
  fileSize: number;
  extractedAt: string;
  lastAccessed: string;
  data: {
    tensors?: any[];
    statistics?: any;
    brainMap?: any;
    config?: any;
    metadata?: any;
  };
}

class IndexedDBCache {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        if (!db.objectStoreNames.contains('models')) {
          const store = db.createObjectStore('models', { keyPath: 'id' });
          store.createIndex('modelId', 'modelId', { unique: false });
          store.createIndex('lastAccessed', 'lastAccessed', { unique: false });
        }
      };
    });
  }

  private getStore(mode: IDBTransactionMode = 'readonly'): IDBObjectStore {
    if (!this.db) {
      throw new Error('Database not initialized. Call init() first.');
    }
    const transaction = this.db.transaction('models', mode);
    return transaction.objectStore('models');
  }

  async save(model: CachedModel): Promise<void> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('readwrite');
      const request = store.put(model);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async get(id: string): Promise<CachedModel | undefined> {
    return new Promise((resolve, reject) => {
      const store = this.getStore();
      const request = store.get(id);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const result = request.result;
        if (result) {
          // Update last accessed time
          result.lastAccessed = new Date().toISOString();
          this.save(result).catch(console.error);
        }
        resolve(result);
      };
    });
  }

  async getByModelId(modelId: string): Promise<CachedModel[]> {
    return new Promise((resolve, reject) => {
      const store = this.getStore();
      const index = store.index('modelId');
      const request = index.getAll(modelId);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  async getAll(): Promise<CachedModel[]> {
    return new Promise((resolve, reject) => {
      const store = this.getStore();
      const request = store.getAll();
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  async delete(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('readwrite');
      const request = store.delete(id);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async clear(): Promise<void> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('readwrite');
      const request = store.clear();
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async getCacheSize(): Promise<number> {
    const models = await this.getAll();
    return models.reduce((size, model) => {
      return size + JSON.stringify(model.data).length;
    }, 0);
  }

  async getOldestModels(limit: number = 5): Promise<CachedModel[]> {
    const models = await this.getAll();
    return models
      .sort((a, b) => new Date(a.lastAccessed).getTime() - new Date(b.lastAccessed).getTime())
      .slice(0, limit);
  }

  async cleanup(maxSizeBytes: number = 500 * 1024 * 1024): Promise<void> {
    // 500MB default limit
    let currentSize = await this.getCacheSize();
    
    if (currentSize <= maxSizeBytes) {
      return;
    }

    const oldest = await this.getOldestModels(10);
    for (const model of oldest) {
      await this.delete(model.id);
      currentSize = await this.getCacheSize();
      if (currentSize <= maxSizeBytes) {
        break;
      }
    }
  }
}

export const cache = new IndexedDBCache();

// Helper functions for common operations
export async function cacheModelExtraction(
  modelId: string,
  fileName: string,
  fileSize: number,
  data: CachedModel['data']
): Promise<string> {
  const id = `${modelId}_${Date.now()}`;
  const model: CachedModel = {
    id,
    modelId,
    fileName,
    fileSize,
    extractedAt: new Date().toISOString(),
    lastAccessed: new Date().toISOString(),
    data,
  };
  
  await cache.save(model);
  await cache.cleanup(); // Auto-cleanup if over limit
  
  return id;
}

export async function getCachedExtraction(id: string): Promise<CachedModel | undefined> {
  return cache.get(id);
}

export async function getCachedByModel(modelId: string): Promise<CachedModel[]> {
  return cache.getByModelId(modelId);
}

export async function clearCache(): Promise<void> {
  await cache.clear();
}

export async function getCacheStats(): Promise<{
  count: number;
  sizeBytes: number;
  sizeMB: number;
}> {
  const models = await cache.getAll();
  const sizeBytes = await cache.getCacheSize();
  
  return {
    count: models.length,
    sizeBytes,
    sizeMB: sizeBytes / (1024 * 1024),
  };
}
