import React, { useState } from 'react';
import { Upload, Database, FileText, BarChart3, CheckCircle, AlertCircle, XCircle, TrendingUp, Shield, PieChart, Activity } from 'lucide-react';

interface DataRow {
  [key: string]: any;
}

interface DataSummary {
  totalRows: number;
  totalColumns: number;
  columnInfo: {
    [key: string]: {
      type: string;
      nullCount: number;
      uniqueCount: number;
      mean?: number;
      median?: number;
      std?: number;
      min?: number;
      max?: number;
    };
  };
}

interface FieldMapping {
  userField: string;
  suggestedField: string;
  dataType: string;
  confidenceScore: number;
  isActive: boolean;
}

interface DatabaseCredentials {
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  dbType: string;
}

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'dashboard' | 'preview' | 'mapping' | 'claims-summary' | 'fwa-detection' | 'trend-analysis'>('dashboard');
  const [uploadedData, setUploadedData] = useState<DataRow[]>([]);
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [showDbModal, setShowDbModal] = useState(false);
  const [fieldMappings, setFieldMappings] = useState<FieldMapping[]>([]);
  const [activeScenarios, setActiveScenarios] = useState<string[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [mappingConfirmed, setMappingConfirmed] = useState(false);

  // Sample field mappings for FWA analytics
  const requiredFields = [
    { field: 'claim_id', dataType: 'string', description: 'Unique claim identifier' },
    { field: 'patient_id', dataType: 'string', description: 'Patient identifier' },
    { field: 'provider_id', dataType: 'string', description: 'Healthcare provider ID' },
    { field: 'claim_amount', dataType: 'number', description: 'Total claim amount' },
    { field: 'service_date', dataType: 'date', description: 'Date of service' },
    { field: 'diagnosis_code', dataType: 'string', description: 'Primary diagnosis code' },
    { field: 'procedure_code', dataType: 'string', description: 'Medical procedure code' },
    { field: 'member_age', dataType: 'number', description: 'Patient age' },
    { field: 'service_location', dataType: 'string', description: 'Place of service' }
  ];

  const fwaScenarios = [
    { 
      name: 'Duplicate Claims Detection', 
      requiredFields: ['claim_id', 'patient_id', 'provider_id', 'claim_amount', 'service_date'],
      description: 'Identifies potential duplicate claims submitted by providers'
    },
    { 
      name: 'Billing Pattern Analysis', 
      requiredFields: ['provider_id', 'claim_amount', 'service_date', 'procedure_code'],
      description: 'Analyzes unusual billing patterns and frequency'
    },
    { 
      name: 'Age-Service Mismatch', 
      requiredFields: ['member_age', 'procedure_code', 'diagnosis_code'],
      description: 'Detects services inappropriate for patient age'
    },
    { 
      name: 'Provider Network Analysis', 
      requiredFields: ['provider_id', 'patient_id', 'service_location'],
      description: 'Identifies suspicious provider-patient relationships'
    },
    { 
      name: 'Amount Outlier Detection', 
      requiredFields: ['claim_amount', 'procedure_code', 'service_location'],
      description: 'Detects claims with unusually high amounts'
    }
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvText = e.target?.result as string;
        const rows = csvText.split('\n');
        const headers = rows[0].split(',').map(h => h.trim().replace(/"/g, ''));
        
        const data: DataRow[] = [];
        for (let i = 1; i < Math.min(rows.length, 101); i++) {
          const values = rows[i].split(',').map(v => v.trim().replace(/"/g, ''));
          if (values.length === headers.length) {
            const row: DataRow = {};
            headers.forEach((header, index) => {
              row[header] = values[index];
            });
            data.push(row);
          }
        }
        
        setUploadedData(data);
        generateDataSummary(data, headers);
        generateFieldMappings(headers);
        setCurrentStep('preview');
      };
      reader.readAsText(file);
    }
  };

  const generateDataSummary = (data: DataRow[], headers: string[]) => {
    const columnInfo: { [key: string]: { type: string; nullCount: number; uniqueCount: number; mean?: number; median?: number; std?: number; min?: number; max?: number } } = {};
    
    headers.forEach(header => {
      const values = data.map(row => row[header]).filter(v => v !== '' && v !== undefined);
      const uniqueValues = new Set(values);
      
      let type = 'string';
      let mean, median, std, min, max;
      
      if (values.every(v => !isNaN(Number(v)) && v !== '')) {
        type = 'number';
        const numValues = values.map(v => Number(v));
        mean = numValues.reduce((a, b) => a + b, 0) / numValues.length;
        min = Math.min(...numValues);
        max = Math.max(...numValues);
        
        const sorted = [...numValues].sort((a, b) => a - b);
        median = sorted.length % 2 === 0 
          ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
          : sorted[Math.floor(sorted.length / 2)];
        
        const variance = numValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / numValues.length;
        std = Math.sqrt(variance);
      } else if (values.some(v => /^\d{4}-\d{2}-\d{2}/.test(v))) {
        type = 'date';
      }
      
      columnInfo[header] = {
        type,
        nullCount: data.length - values.length,
        uniqueCount: uniqueValues.size,
        mean: mean ? Number(mean.toFixed(2)) : undefined,
        median: median ? Number(median.toFixed(2)) : undefined,
        std: std ? Number(std.toFixed(2)) : undefined,
        min,
        max
      };
    });
    
    setDataSummary({
      totalRows: data.length,
      totalColumns: headers.length,
      columnInfo
    });
  };

  const generateFieldMappings = (userFields: string[]) => {
    const mappings: FieldMapping[] = requiredFields.map(required => {
      const matches = userFields.map(userField => {
        const similarity = calculateSimilarity(userField.toLowerCase(), required.field.toLowerCase());
        return { userField, similarity };
      });
      
      const bestMatch = matches.reduce((prev, current) => 
        (current.similarity > prev.similarity) ? current : prev
      );
      
      const confidenceScore = Math.round(bestMatch.similarity * 100);
      
      return {
        userField: bestMatch.userField,
        suggestedField: required.field,
        dataType: required.dataType,
        confidenceScore,
        isActive: confidenceScore >= 70
      };
    });
    
    setFieldMappings(mappings);
  };

  const calculateSimilarity = (str1: string, str2: string): number => {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();
    
    if (s1.includes(s2) || s2.includes(s1)) return 0.9;
    
    let matches = 0;
    const words1 = s1.split(/[_\s]+/);
    const words2 = s2.split(/[_\s]+/);
    
    words1.forEach(word1 => {
      words2.forEach(word2 => {
        if (word1 === word2 || word1.includes(word2) || word2.includes(word1)) {
          matches++;
        }
      });
    });
    
    return matches / Math.max(words1.length, words2.length);
  };

  const getConfidenceColor = (score: number): string => {
    if (score >= 91) return 'text-green-600 bg-green-100';
    if (score >= 41) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceIcon = (score: number) => {
    if (score >= 91) return <CheckCircle className="h-5 w-5 text-green-600" />;
    if (score >= 41) return <AlertCircle className="h-5 w-5 text-yellow-600" />;
    return <XCircle className="h-5 w-5 text-red-600" />;
  };

  const handleConfirmMapping = () => {
    const activeMappedFields = fieldMappings
      .filter(mapping => mapping.isActive)
      .map(mapping => mapping.suggestedField);
    
    const availableScenarios = fwaScenarios.filter(scenario =>
      scenario.requiredFields.every(field => activeMappedFields.includes(field))
    );
    
    setActiveScenarios(availableScenarios.map(s => s.name));
    setMappingConfirmed(true);
    setCurrentStep('dashboard');
  };

  const DatabaseModal: React.FC = () => {
    const [credentials, setCredentials] = useState<DatabaseCredentials>({
      host: '',
      port: '5432',
      database: '',
      username: '',
      password: '',
      dbType: 'postgresql'
    });

    const handleConnect = () => {
      const sampleData = [
        { claim_id: 'CLM001', patient_id: 'PAT001', provider_id: 'PROV001', claim_amount: '1500.00', service_date: '2024-01-15', diagnosis_code: 'M79.3', procedure_code: '99213', member_age: '45', service_location: 'Office' },
        { claim_id: 'CLM002', patient_id: 'PAT002', provider_id: 'PROV002', claim_amount: '2300.50', service_date: '2024-01-16', diagnosis_code: 'E11.9', procedure_code: '99214', member_age: '62', service_location: 'Office' },
        { claim_id: 'CLM003', patient_id: 'PAT003', provider_id: 'PROV001', claim_amount: '875.25', service_date: '2024-01-17', diagnosis_code: 'I10', procedure_code: '99212', member_age: '38', service_location: 'Office' },
        { claim_id: 'CLM004', patient_id: 'PAT004', provider_id: 'PROV003', claim_amount: '3200.00', service_date: '2024-01-18', diagnosis_code: 'Z00.00', procedure_code: '99215', member_age: '29', service_location: 'Office' },
        { claim_id: 'CLM005', patient_id: 'PAT005', provider_id: 'PROV002', claim_amount: '1890.75', service_date: '2024-01-19', diagnosis_code: 'M25.511', procedure_code: '99213', member_age: '55', service_location: 'Office' }
      ];
      
      setUploadedData(sampleData);
      generateDataSummary(sampleData, Object.keys(sampleData[0]));
      generateFieldMappings(Object.keys(sampleData[0]));
      setShowDbModal(false);
      setCurrentStep('preview');
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4">
          <h3 className="text-xl font-bold mb-6">Database Connection</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Database Type</label>
              <select 
                value={credentials.dbType} 
                onChange={(e) => setCredentials(prev => ({ ...prev, dbType: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="sqlserver">SQL Server</option>
                <option value="oracle">Oracle</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Host</label>
              <input 
                type="text" 
                value={credentials.host}
                onChange={(e) => setCredentials(prev => ({ ...prev, host: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
                placeholder="localhost"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Port</label>
              <input 
                type="text" 
                value={credentials.port}
                onChange={(e) => setCredentials(prev => ({ ...prev, port: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Database Name</label>
              <input 
                type="text" 
                value={credentials.database}
                onChange={(e) => setCredentials(prev => ({ ...prev, database: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
              <input 
                type="text" 
                value={credentials.username}
                onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input 
                type="password" 
                value={credentials.password}
                onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
          </div>
          
          <div className="flex gap-3 mt-6">
            <button 
              onClick={handleConnect}
              className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Connect
            </button>
            <button 
              onClick={() => setShowDbModal(false)}
              className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-400 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  };

  if (currentStep === 'dashboard') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-6 py-8">
          {/* Header with Upload Options */}
          <div className="flex justify-between items-start mb-12">
            <div>
              <h1 className="text-4xl font-bold text-gray-800 mb-4">Healthcare Analytics Tool</h1>
              <p className="text-lg text-gray-600 max-w-2xl">
                Upload healthcare claims data to perform comprehensive analytics on claims datasets. 
                Access different analytical modules including FWA (Fraud, Waste, and Abuse) analytics.
              </p>
            </div>
            
            {/* Upload Options - Top Right */}
            <div className="flex gap-4">
              <label className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors cursor-pointer inline-flex items-center gap-2">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Upload className="h-5 w-5" />
                Upload File
              </label>
              
              <button 
                onClick={() => setShowDbModal(true)}
                className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors inline-flex items-center gap-2"
              >
                <Database className="h-5 w-5" />
                Connect Your DB
              </button>
            </div>
          </div>

          {/* Analytics Modules */}
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Claims Data Summary */}
            <div className={`bg-white rounded-xl shadow-lg p-8 transition-all ${mappingConfirmed ? 'hover:shadow-xl cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}
                 onClick={() => mappingConfirmed && setCurrentStep('claims-summary')}>
              <div className="text-center">
                <PieChart className={`h-16 w-16 mx-auto mb-4 ${mappingConfirmed ? 'text-blue-600' : 'text-gray-400'}`} />
                <h3 className="text-2xl font-semibold text-gray-800 mb-4">Claims Data Summary</h3>
                <p className="text-gray-600 mb-6">
                  Comprehensive statistical summary of all available fields with visualizations and insights
                </p>
                {!mappingConfirmed && (
                  <p className="text-sm text-red-500">Complete field mapping first</p>
                )}
              </div>
            </div>

            {/* FWA Detection */}
            <div className={`bg-white rounded-xl shadow-lg p-8 transition-all ${mappingConfirmed ? 'hover:shadow-xl cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}
                 onClick={() => mappingConfirmed && setCurrentStep('fwa-detection')}>
              <div className="text-center">
                <Shield className={`h-16 w-16 mx-auto mb-4 ${mappingConfirmed ? 'text-red-600' : 'text-gray-400'}`} />
                <h3 className="text-2xl font-semibold text-gray-800 mb-4">FWA Detection</h3>
                <p className="text-gray-600 mb-6">
                  Advanced fraud, waste, and abuse detection using multiple analytical scenarios
                </p>
                {!mappingConfirmed && (
                  <p className="text-sm text-red-500">Complete field mapping first</p>
                )}
              </div>
            </div>

            {/* Trend Analysis */}
            <div className={`bg-white rounded-xl shadow-lg p-8 transition-all ${mappingConfirmed ? 'hover:shadow-xl cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}
                 onClick={() => mappingConfirmed && setCurrentStep('trend-analysis')}>
              <div className="text-center">
                <TrendingUp className={`h-16 w-16 mx-auto mb-4 ${mappingConfirmed ? 'text-green-600' : 'text-gray-400'}`} />
                <h3 className="text-2xl font-semibold text-gray-800 mb-4">Trend Analysis</h3>
                <p className="text-gray-600 mb-6">
                  Identify patterns and trends in healthcare claims data over time
                </p>
                {!mappingConfirmed && (
                  <p className="text-sm text-red-500">Complete field mapping first</p>
                )}
              </div>
            </div>
          </div>

          {/* Field Mapping Button */}
          {uploadedData.length > 0 && !mappingConfirmed && (
            <div className="text-center mt-12">
              <button 
                onClick={() => setCurrentStep('mapping')}
                className="bg-purple-600 text-white px-8 py-3 rounded-lg hover:bg-purple-700 transition-colors text-lg font-medium"
              >
                Field Mapping
              </button>
            </div>
          )}
        </div>

        {showDbModal && <DatabaseModal />}
      </div>
    );
  }

  if (currentStep === 'preview') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-6">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Data Preview & Summary</h2>
            
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Data Preview */}
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="h-5 w-5 text-blue-600" />
                  <h3 className="text-xl font-semibold">Data Preview (First 5 rows)</h3>
                </div>
                
                <div className="overflow-x-auto border rounded-lg">
                  <table className="min-w-full bg-white">
                    <thead className="bg-gray-50">
                      <tr>
                        {uploadedData.length > 0 && Object.keys(uploadedData[0]).map(key => (
                          <th key={key} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {uploadedData.slice(0, 5).map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, colIndex) => (
                            <td key={colIndex} className="px-4 py-3 text-sm text-gray-600">
                              {String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Data Summary */}
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <BarChart3 className="h-5 w-5 text-green-600" />
                  <h3 className="text-xl font-semibold">Data Summary</h3>
                </div>
                
                {dataSummary && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{dataSummary.totalRows}</div>
                        <div className="text-sm text-gray-600">Total Rows</div>
                      </div>
                      <div className="bg-green-50 p-4 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{dataSummary.totalColumns}</div>
                        <div className="text-sm text-gray-600">Total Columns</div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-3">Column Information</h4>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {Object.entries(dataSummary.columnInfo).map(([column, info]) => (
                          <div key={column} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                            <div>
                              <div className="font-medium">{column}</div>
                              <div className="text-sm text-gray-500">Type: {info.type}</div>
                            </div>
                            <div className="text-right">
                              <div className="text-sm">Unique: {info.uniqueCount}</div>
                              <div className="text-sm text-red-500">Nulls: {info.nullCount}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <div className="mt-8 text-center">
              <button 
                onClick={() => setCurrentStep('mapping')}
                className="bg-purple-600 text-white px-8 py-3 rounded-lg hover:bg-purple-700 transition-colors text-lg font-medium"
              >
                Field Mapping
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === 'mapping') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-6">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Field Mapping</h2>
            <p className="text-gray-600 mb-8">Required to run FWA Analytics</p>
            
            <div className="space-y-4">
              <div className="grid grid-cols-5 gap-4 font-medium text-gray-700 border-b pb-3">
                <div>Required Field</div>
                <div>Your Data Field</div>
                <div>Data Type</div>
                <div>Confidence Score</div>
                <div>Action</div>
              </div>
              
              {fieldMappings.map((mapping, index) => (
                <div key={index} className="grid grid-cols-5 gap-4 items-center py-3 border-b">
                  <div className="font-medium">{mapping.suggestedField}</div>
                  
                  <div>
                    <select 
                      value={mapping.userField}
                      onChange={(e) => {
                        const newMappings = [...fieldMappings];
                        newMappings[index].userField = e.target.value;
                        setFieldMappings(newMappings);
                      }}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                    >
                      {uploadedData.length > 0 && Object.keys(uploadedData[0]).map(field => (
                        <option key={field} value={field}>{field}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="text-sm text-gray-600">{mapping.dataType}</div>
                  
                  <div className="flex items-center gap-2">
                    {getConfidenceIcon(mapping.confidenceScore)}
                    <span className={`px-2 py-1 rounded text-sm font-medium ${getConfidenceColor(mapping.confidenceScore)}`}>
                      {mapping.confidenceScore}%
                    </span>
                  </div>
                  
                  <div>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={mapping.isActive}
                        onChange={(e) => {
                          const newMappings = [...fieldMappings];
                          newMappings[index].isActive = e.target.checked;
                          setFieldMappings(newMappings);
                        }}
                        className="rounded"
                      />
                      <span className="ml-2 text-sm">Active</span>
                    </label>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-8 flex justify-center gap-4">
              <button 
                onClick={handleConfirmMapping}
                className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors text-lg font-medium"
              >
                Confirm Mapping
              </button>
              <button 
                onClick={() => {
                  const newMappings = fieldMappings.map(mapping => ({ ...mapping, isActive: true }));
                  setFieldMappings(newMappings);
                  handleConfirmMapping();
                }}
                className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors text-lg font-medium"
              >
                Confirm All
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === 'claims-summary') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-6">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Claims Data Summary</h2>
              <button 
                onClick={() => setCurrentStep('dashboard')}
                className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Back to Dashboard
              </button>
            </div>
            
            {dataSummary && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-blue-50 p-6 rounded-lg">
                    <div className="text-3xl font-bold text-blue-600">{dataSummary.totalRows}</div>
                    <div className="text-lg text-gray-600">Total Claims</div>
                  </div>
                  <div className="bg-green-50 p-6 rounded-lg">
                    <div className="text-3xl font-bold text-green-600">{dataSummary.totalColumns}</div>
                    <div className="text-lg text-gray-600">Data Fields</div>
                  </div>
                </div>
                
                <div className="grid lg:grid-cols-2 gap-6">
                  {Object.entries(dataSummary.columnInfo).map(([column, info]) => (
                    <div key={column} className="border rounded-lg p-6">
                      <h3 className="text-lg font-semibold mb-4 text-gray-800">{column}</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Data Type:</span>
                          <span className="font-medium">{info.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Unique Values:</span>
                          <span className="font-medium">{info.uniqueCount}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Null Count:</span>
                          <span className="font-medium text-red-600">{info.nullCount}</span>
                        </div>
                        {info.type === 'number' && (
                          <>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Mean:</span>
                              <span className="font-medium">{info.mean}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Median:</span>
                              <span className="font-medium">{info.median}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Std Dev:</span>
                              <span className="font-medium">{info.std}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Min:</span>
                              <span className="font-medium">{info.min}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Max:</span>
                              <span className="font-medium">{info.max}</span>
                            </div>
                          </>
                        )}
                      </div>
                      
                      {/* Simple visualization placeholder */}
                      <div className="mt-4 h-20 bg-gradient-to-r from-blue-100 to-blue-200 rounded flex items-center justify-center">
                        <Activity className="h-8 w-8 text-blue-600" />
                        <span className="ml-2 text-blue-700">Distribution Chart</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === 'fwa-detection') {
    const availableScenarios = fwaScenarios.filter(scenario => 
      activeScenarios.includes(scenario.name)
    );
    
    const unavailableScenarios = fwaScenarios.filter(scenario => 
      !activeScenarios.includes(scenario.name)
    );

    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-6">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">FWA Detection Scenarios</h2>
              <button 
                onClick={() => setCurrentStep('dashboard')}
                className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Back to Dashboard
              </button>
            </div>
            
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Available Scenarios */}
              <div>
                <h3 className="text-xl font-semibold text-green-600 mb-4 flex items-center gap-2">
                  <CheckCircle className="h-6 w-6" />
                  Available Scenarios ({availableScenarios.length})
                </h3>
                
                <div className="space-y-4">
                  {availableScenarios.map((scenario, index) => (
                    <div key={index} className="border border-green-200 bg-green-50 rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <input
                          type="radio"
                          name="scenario"
                          value={scenario.name}
                          checked={selectedScenarios.includes(scenario.name)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedScenarios([...selectedScenarios, scenario.name]);
                            } else {
                              setSelectedScenarios(selectedScenarios.filter(s => s !== scenario.name));
                            }
                          }}
                          className="mt-1"
                        />
                        <div className="flex-1">
                          <h4 className="font-medium text-green-800 mb-2">{scenario.name}</h4>
                          <p className="text-sm text-green-700 mb-3">{scenario.description}</p>
                          <div className="text-xs text-green-600">
                            Required fields: {scenario.requiredFields.join(', ')}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Unavailable Scenarios */}
              <div>
                <h3 className="text-xl font-semibold text-red-600 mb-4 flex items-center gap-2">
                  <XCircle className="h-6 w-6" />
                  Unavailable Scenarios ({unavailableScenarios.length})
                </h3>
                
                <div className="space-y-4">
                  {unavailableScenarios.map((scenario, index) => (
                    <div key={index} className="border border-red-200 bg-red-50 rounded-lg p-4">
                      <h4 className="font-medium text-red-800 mb-2">{scenario.name}</h4>
                      <p className="text-sm text-red-700 mb-3">{scenario.description}</p>
                      <div className="text-xs text-red-600">
                        Missing fields: {scenario.requiredFields.filter(field => 
                          !fieldMappings.some(mapping => mapping.suggestedField === field && mapping.isActive)
                        ).join(', ')}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="mt-8 text-center">
              <button 
                className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-12 py-4 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all text-lg font-medium shadow-lg disabled:opacity-50"
                disabled={selectedScenarios.length === 0}
                onClick={() => alert(`Running selected FWA Analytics scenarios: ${selectedScenarios.join(', ')}. This would process your data and generate fraud detection reports.`)}
              >
                Run Selected Scenarios ({selectedScenarios.length})
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === 'trend-analysis') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-6">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Trend Analysis</h2>
              <button 
                onClick={() => setCurrentStep('dashboard')}
                className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Back to Dashboard
              </button>
            </div>
            
            <div className="text-center py-20">
              <TrendingUp className="h-24 w-24 text-green-600 mx-auto mb-6" />
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">Trend Analysis Module</h3>
              <p className="text-gray-600 max-w-2xl mx-auto">
                This module will analyze temporal patterns in your healthcare claims data, 
                identifying trends in claim volumes, amounts, provider behaviors, and seasonal variations.
              </p>
              <button className="mt-8 bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors">
                Start Trend Analysis
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default App;