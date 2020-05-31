using System;
using System.Net.Http;
using LaYumba.Functional;

namespace KnnIris
{
    public class CsvDownloader
    {
        private readonly HttpClient _client;

        public CsvDownloader()
        {
            _client = new HttpClient();
        }

        public Either<NetworkRequestError, string> DownloadFromUrl(Uri uri)
        {
            try
            {
                var result = _client.GetAsync(uri).Result;
                if (result.IsSuccessStatusCode) return result.Content.ReadAsStringAsync().Result;
                else return new NetworkRequestError("status code: " + result.StatusCode);
            }
            catch (ArgumentException e)
            {
                Console.WriteLine(e);
                return new NetworkRequestError("url cannot be null");
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine(e);
                return new NetworkRequestError(e.Message);
            }
        }
    }

    public sealed class NetworkRequestError : Error
    {
        public override string Message { get; }

        public NetworkRequestError(string message)
        {
            Message = message;
        }
    }
}